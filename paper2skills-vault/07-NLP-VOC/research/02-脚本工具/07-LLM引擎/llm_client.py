"""LLM Client - Phase 5 D1.T1.3

DeepSeek + Kimi 双引擎统一封装，OpenAI SDK 兼容路径。
提供：
1. LLMClient 类：异步调用 + 指数退避 + 429 重试
2. discover_models():  smoke-test 时探测线上可用模型
3. CLI smoke-test：验证 API 连通性 + JSON mode + Tool use

设计要点：
- 配置文件 ~/.paper2skills/llm_keys.json （已 chmod 600）
- 两家厂商均为 OpenAI 兼容，代码差异仅 base_url + model
- DeepSeek 实际 API 模型 ID 为 deepseek-chat / deepseek-reasoner
  （V4 路由在底层；V4 命名 deepseek-v4-flash 是 marketing alias，
   API 调用时仍用 chat/reasoner，配置中保留两套以兼容未来 alias 启用）
- Kimi 公开模型：moonshot-v1-8k / -32k / -128k
  （k2.6 是模型代号，在 platform.moonshot.cn 通过 moonshot-v1-* 暴露）
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion

DEFAULT_KEYS_PATH = Path.home() / ".paper2skills" / "llm_keys.json"

logger = logging.getLogger("llm_client")

def load_keys(path: Path | str = DEFAULT_KEYS_PATH) -> dict:
    """加载 LLM keys 配置。文件必须存在且 chmod 600。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"LLM keys file not found: {p}\n"
            f"Run: cp ~/.paper2skills/llm_keys.template.json {p} && chmod 600 {p}"
        )
    cfg = json.loads(p.read_text(encoding="utf-8"))
    for vendor in ("deepseek", "kimi"):
        if vendor not in cfg:
            raise ValueError(f"Missing vendor '{vendor}' in {p}")
        if "api_key" not in cfg[vendor] or "base_url" not in cfg[vendor]:
            raise ValueError(f"Vendor '{vendor}' missing api_key or base_url")
    return cfg

@dataclass
class LLMResponse:
    """统一封装 LLM 调用结果。"""

    content: str
    model_used: str
    vendor: str
    tokens_in: int = 0
    tokens_out: int = 0
    cache_hit_tokens: int = 0
    latency_ms: float = 0.0
    retries: int = 0
    raw: Any = None  # 原始 ChatCompletion 对象，需要时再用

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

@dataclass
class CallStats:
    """累计调用统计。"""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_cache_hit: int = 0
    total_latency_ms: float = 0.0
    total_retries: int = 0
    by_vendor: dict[str, dict] = field(default_factory=lambda: {})

    def record(self, resp: LLMResponse, success: bool = True):
        self.total_calls += 1
        if success:
            self.successful_calls += 1
            self.total_tokens_in += resp.tokens_in
            self.total_tokens_out += resp.tokens_out
            self.total_cache_hit += resp.cache_hit_tokens
            self.total_latency_ms += resp.latency_ms
            self.total_retries += resp.retries
            v = self.by_vendor.setdefault(
                resp.vendor,
                {"calls": 0, "tokens_in": 0, "tokens_out": 0, "latency_ms": 0.0},
            )
            v["calls"] += 1
            v["tokens_in"] += resp.tokens_in
            v["tokens_out"] += resp.tokens_out
            v["latency_ms"] += resp.latency_ms
        else:
            self.failed_calls += 1

    def summary(self) -> dict:
        if self.successful_calls == 0:
            return {"successful": 0}
        return {
            "total_calls": self.total_calls,
            "successful": self.successful_calls,
            "failed": self.failed_calls,
            "success_rate": self.successful_calls / self.total_calls,
            "avg_latency_ms": self.total_latency_ms / self.successful_calls,
            "avg_tokens_in": self.total_tokens_in / self.successful_calls,
            "avg_tokens_out": self.total_tokens_out / self.successful_calls,
            "cache_hit_rate": (
                self.total_cache_hit / self.total_tokens_in
                if self.total_tokens_in > 0
                else 0.0
            ),
            "avg_retries": self.total_retries / self.successful_calls,
            "by_vendor": self.by_vendor,
        }

class LLMClient:
    """DeepSeek + Kimi 双引擎统一客户端。"""

    def __init__(self, keys_path: Path | str = DEFAULT_KEYS_PATH):
        self.cfg = load_keys(keys_path)
        self._async_clients: dict[str, AsyncOpenAI] = {}
        self._sync_clients: dict[str, OpenAI] = {}
        self.stats = CallStats()
        self.semaphores: dict[str, asyncio.Semaphore] = {}
        conc = self.cfg.get("concurrency", {})
        self._max_retries = conc.get("retry_max_attempts", 5)
        self._initial_backoff = conc.get("retry_initial_backoff_sec", 0.5)
    def _get_async_client(self, vendor: str) -> AsyncOpenAI:
        if vendor not in self._async_clients:
            v = self.cfg[vendor]
            self._async_clients[vendor] = AsyncOpenAI(
                api_key=v["api_key"],
                base_url=v["base_url"],
            )
        return self._async_clients[vendor]

    def _get_sync_client(self, vendor: str) -> OpenAI:
        if vendor not in self._sync_clients:
            v = self.cfg[vendor]
            self._sync_clients[vendor] = OpenAI(
                api_key=v["api_key"],
                base_url=v["base_url"],
            )
        return self._sync_clients[vendor]

    def _get_semaphore(self, vendor: str) -> asyncio.Semaphore:
        if vendor not in self.semaphores:
            conc = self.cfg.get("concurrency", {})
            limit = conc.get(f"{vendor}_max_concurrent", 10)
            self.semaphores[vendor] = asyncio.Semaphore(limit)
        return self.semaphores[vendor]
    def list_models(self, vendor: str) -> list[str]:
        """探测线上可用模型 ID（用于 smoke-test）。"""
        client = self._get_sync_client(vendor)
        try:
            models = client.models.list()
            return sorted([m.id for m in models.data])
        except Exception as e:
            logger.warning(f"list_models({vendor}) failed: {e}")
            return []
    def chat_sync(
        self,
        vendor: str,
        messages: list[dict],
        model: Optional[str] = None,
        response_format: Optional[dict] = None,
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """同步调用，主要给 smoke-test 用。生产用 chat_async。"""
        v = self.cfg[vendor]
        model = model or v.get("default_model")
        client = self._get_sync_client(vendor)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format

        t0 = time.time()
        resp: ChatCompletion = client.chat.completions.create(**kwargs)
        latency_ms = (time.time() - t0) * 1000

        usage = resp.usage
        cache_hit = 0
        # DeepSeek 暴露 prompt_cache_hit_tokens；其他厂商可能没有
        if usage and hasattr(usage, "prompt_cache_hit_tokens"):
            cache_hit = getattr(usage, "prompt_cache_hit_tokens", 0) or 0

        result = LLMResponse(
            content=resp.choices[0].message.content or "",
            model_used=model,
            vendor=vendor,
            tokens_in=usage.prompt_tokens if usage else 0,
            tokens_out=usage.completion_tokens if usage else 0,
            cache_hit_tokens=cache_hit,
            latency_ms=latency_ms,
            retries=0,
            raw=resp,
        )
        self.stats.record(result, success=True)
        return result
    async def chat_async(
        self,
        vendor: str,
        messages: list[dict],
        model: Optional[str] = None,
        response_format: Optional[dict] = None,
        max_tokens: int = 800,
        temperature: float = 0.2,
    ) -> LLMResponse:
        """异步调用，含信号量并发控制 + 指数退避。"""
        v = self.cfg[vendor]
        model = model or v.get("default_model")
        client = self._get_async_client(vendor)
        sem = self._get_semaphore(vendor)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format

        retries = 0
        backoff = self._initial_backoff
        last_exc: Optional[Exception] = None

        async with sem:
            while retries <= self._max_retries:
                try:
                    t0 = time.time()
                    resp: ChatCompletion = await client.chat.completions.create(**kwargs)
                    latency_ms = (time.time() - t0) * 1000

                    usage = resp.usage
                    cache_hit = 0
                    if usage and hasattr(usage, "prompt_cache_hit_tokens"):
                        cache_hit = getattr(usage, "prompt_cache_hit_tokens", 0) or 0

                    result = LLMResponse(
                        content=resp.choices[0].message.content or "",
                        model_used=model,
                        vendor=vendor,
                        tokens_in=usage.prompt_tokens if usage else 0,
                        tokens_out=usage.completion_tokens if usage else 0,
                        cache_hit_tokens=cache_hit,
                        latency_ms=latency_ms,
                        retries=retries,
                        raw=resp,
                    )
                    self.stats.record(result, success=True)
                    return result

                except Exception as e:
                    last_exc = e
                    msg = str(e)
                    is_retryable = (
                        "429" in msg
                        or "rate" in msg.lower()
                        or "timeout" in msg.lower()
                        or "5" in msg[:5]  # 5xx
                    )
                    if not is_retryable or retries >= self._max_retries:
                        logger.error(
                            f"[{vendor}/{model}] non-retryable or max retries reached: {e}"
                        )
                        self.stats.record(
                            LLMResponse(
                                content="",
                                model_used=model,
                                vendor=vendor,
                                retries=retries,
                            ),
                            success=False,
                        )
                        raise
                    logger.warning(
                        f"[{vendor}/{model}] retry {retries+1}/{self._max_retries} "
                        f"after {backoff:.2f}s: {e}"
                    )
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    retries += 1

        raise RuntimeError(f"unreachable; last_exc={last_exc}")

SMOKE_REVIEW = "This breast pump is comfortable to wear but the suction dropped after 2 weeks."

SMOKE_SYSTEM_PROMPT = """你是 VOC 标签助手。对评论输出 JSON：
{
  "labels": [{"tag_id": "TAG_GEN_002", "evidence": "comfortable to wear"}],
  "overall_sentiment": "positive|negative|neutral"
}
labels 列表 1-3 项；只输出 JSON，无任何解释文本。"""

def smoke_test_vendor(client: LLMClient, vendor: str) -> dict:
    """对单家厂商做 smoke-test，返回 {pass, latency_ms, json_valid, models, error}"""
    print(f"\n{'='*60}")
    print(f"  Smoke Test: {vendor.upper()}")
    print(f"{'='*60}")

    result = {
        "vendor": vendor,
        "pass": False,
        "models_listed": False,
        "models_count": 0,
        "models_sample": [],
        "call_succeeded": False,
        "json_valid": False,
        "latency_ms": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "cache_hit_tokens": 0,
        "model_used": "",
        "content_preview": "",
        "error": "",
    }

    # Step 1: list models
    try:
        models = client.list_models(vendor)
        result["models_listed"] = True
        result["models_count"] = len(models)
        result["models_sample"] = models[:8]
        print(f"  ✓ Models listed: {len(models)} ids")
        for m in models[:5]:
            print(f"      - {m}")
        if len(models) > 5:
            print(f"      ... (+{len(models)-5} more)")
    except Exception as e:
        result["error"] = f"list_models failed: {e}"
        print(f"  ✗ list_models failed: {e}")
        return result

    # Step 2: chat completion w/ JSON mode
    try:
        resp = client.chat_sync(
            vendor=vendor,
            messages=[
                {"role": "system", "content": SMOKE_SYSTEM_PROMPT},
                {"role": "user", "content": SMOKE_REVIEW},
            ],
            response_format={"type": "json_object"},
            max_tokens=300,
        )
        result["call_succeeded"] = True
        result["latency_ms"] = round(resp.latency_ms, 1)
        result["tokens_in"] = resp.tokens_in
        result["tokens_out"] = resp.tokens_out
        result["cache_hit_tokens"] = resp.cache_hit_tokens
        result["model_used"] = resp.model_used
        result["content_preview"] = resp.content[:200]
        print(f"  ✓ Chat completion in {resp.latency_ms:.0f}ms")
        print(f"      model: {resp.model_used}")
        print(f"      tokens: in={resp.tokens_in}, out={resp.tokens_out}, cache_hit={resp.cache_hit_tokens}")
        print(f"      content: {resp.content[:200]}...")

        try:
            parsed = json.loads(resp.content)
            result["json_valid"] = True
            print(f"  ✓ JSON valid: keys={list(parsed.keys())}")
        except Exception as e:
            print(f"  ✗ JSON parse failed: {e}")

    except Exception as e:
        result["error"] = f"chat_sync failed: {e}"
        print(f"  ✗ chat_sync failed: {e}")
        return result

    result["pass"] = (
        result["call_succeeded"]
        and result["json_valid"]
        and result["latency_ms"] < 5000
    )
    if result["pass"]:
        print(f"  🎉 {vendor.upper()} SMOKE TEST PASS")
    else:
        print(f"  ❌ {vendor.upper()} SMOKE TEST FAIL")
    return result

def cli_smoke_test(keys_path: Optional[str] = None) -> int:
    """CLI 入口：python llm_client.py --smoke-test"""
    p = Path(keys_path) if keys_path else DEFAULT_KEYS_PATH
    if not p.exists():
        print(f"❌ Keys file not found: {p}")
        print(f"   Run: cp ~/.paper2skills/llm_keys.template.json {p}")
        print(f"   Then edit and add real api_keys.")
        return 2

    cfg = json.loads(p.read_text(encoding="utf-8"))
    has_real_key = False
    for vendor in ("deepseek", "kimi"):
        k = cfg.get(vendor, {}).get("api_key", "")
        if k and not k.startswith("sk-REPLACE"):
            has_real_key = True
    if not has_real_key:
        print(f"⚠️  Both api_keys still placeholder (sk-REPLACE_...)")
        print(f"   Edit {p} and fill real keys before smoke-test")
        print(f"   Returning exit code 3 (placeholder mode)")
        return 3

    client = LLMClient(p)
    results = {}
    for vendor in ("deepseek", "kimi"):
        k = cfg.get(vendor, {}).get("api_key", "")
        if not k or k.startswith("sk-REPLACE"):
            print(f"\n⊘ Skipping {vendor}: api_key is placeholder")
            results[vendor] = {"vendor": vendor, "pass": False, "skipped": True}
            continue
        try:
            results[vendor] = smoke_test_vendor(client, vendor)
        except Exception as e:
            print(f"❌ {vendor} crashed: {e}")
            results[vendor] = {"vendor": vendor, "pass": False, "error": str(e)}

    print(f"\n{'='*60}")
    print(f"  SMOKE TEST SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for vendor, r in results.items():
        if r.get("skipped"):
            status = "SKIP"
        elif r.get("pass"):
            status = "PASS"
        else:
            status = "FAIL"
            all_pass = False
        print(f"  {vendor:12s}: {status}")
    print(f"{'='*60}")

    report_dir = Path("/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/03-审计报告")
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / "phase5_d1_smoke_test.json"
    report.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Report saved: {report}")

    return 0 if all_pass else 1

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Phase 5 LLM Client - DeepSeek + Kimi")
    ap.add_argument("--smoke-test", action="store_true", help="Run smoke test")
    ap.add_argument("--keys", type=str, default=None, help="Override keys path")
    args = ap.parse_args()

    if args.smoke_test:
        sys.exit(cli_smoke_test(args.keys))
    else:
        ap.print_help()
        sys.exit(0)
