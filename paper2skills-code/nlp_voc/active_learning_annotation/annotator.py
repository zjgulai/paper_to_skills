"""LLM-as-Annotator 核心实现

生产环境支持真实 LLM API 调用（OpenAI/DeepSeek/兼容接口），
默认保持 simulate 模式用于本地测试和离线开发。
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Optional

# ── 可选依赖：openai SDK ──────────────────────────────────────
try:
    import openai

    _HAS_OPENAI = True
except ImportError:
    openai = None  # type: ignore
    _HAS_OPENAI = False

logger = logging.getLogger(__name__)


@dataclass
class AnnotationResult:
    """单条标注结果"""

    text: str
    label: str
    confidence: str  # "high" | "medium" | "low"
    reasoning: str = ""  # LLM 的标注理由
    model_name: str = "simulated"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "label": self.label,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "model": self.model_name,
        }


class LLMAnnotator:
    """LLM 标注器

    支持两种模式:
        - simulate=True (默认): 基于关键词规则 + 随机噪声的本地模拟
        - simulate=False: 调用真实 LLM API（OpenAI/DeepSeek 等兼容接口）

    生产环境配置（环境变量）:
        OPENAI_API_KEY    : API 密钥
        OPENAI_BASE_URL   : 自定义 API 基地址（如 DeepSeek）
        OPENAI_MODEL      : 模型名称，默认 gpt-4o-mini
    """

    def __init__(
        self,
        labels: list[str],
        prompt_template: Optional[str] = None,
        few_shot_examples: Optional[list[dict]] = None,
        simulate: bool = True,
        simulate_accuracy: float = 0.85,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
        timeout: float = 30.0,
        max_tokens: int = 64,
    ):
        self.labels = labels
        self.few_shot_examples = few_shot_examples or []
        self.simulate = simulate
        self.simulate_accuracy = simulate_accuracy

        # API 配置
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_tokens = max_tokens

        # 默认 prompt 模板
        self.prompt_template = prompt_template or self._default_prompt()

        # 模拟用：关键词到标签的映射
        self._keyword_map: dict[str, str] = {}
        self._build_keyword_map()

        # 初始化真实 API 客户端
        self._client: Optional[object] = None
        if not simulate:
            if not _HAS_OPENAI:
                raise ImportError(
                    "非模拟模式需要 openai SDK。请安装: pip install openai"
                )
            if not self.api_key:
                raise ValueError(
                    "非模拟模式需要提供 api_key 参数或设置 OPENAI_API_KEY 环境变量"
                )
            client_kwargs: dict = {"timeout": self.timeout}
            if self.api_key:
                client_kwargs["api_key"] = self.api_key
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._client = openai.OpenAI(**client_kwargs)
            logger.info(f"LLMAnnotator 已接入模型: {self.model}")

    def _default_prompt(self) -> str:
        """生成默认标注 prompt"""
        examples_text = ""
        if self.few_shot_examples:
            for ex in self.few_shot_examples[:5]:
                examples_text += f'评论: "{ex["text"]}"\n标签: {ex["label"]}\n\n'

        return f"""你是一名电商评论标注专家。请将以下母婴产品评论分类到给定标签中。

可用标签: {', '.join(self.labels)}

{examples_text}请直接输出标签名称，不要解释。
评论: "{{text}}"
标签:"""

    def _build_keyword_map(self) -> None:
        """构建关键词到标签的映射（仅模拟模式使用）"""
        # 为每个标签定义触发关键词
        keyword_patterns = {
            "尺码偏差": ["尺码", "大小", "偏大", "偏小", "不合身", "sizing", "size"],
            "材质问题": ["材质", "面料", "硬", "粗糙", "不舒服", "material", "fabric"],
            "漏尿": ["漏", "漏尿", "leak", "leaking", "渗漏"],
            "腰贴问题": ["腰贴", "魔术贴", "粘", "tab", "waistband"],
            "物流延迟": ["物流", "快递", "慢", "延迟", "shipping", "delivery"],
            "过敏反应": ["过敏", "红疹", "发红", "allergy", "rash", "red"],
            "价格问题": ["贵", "便宜", "价格", "expensive", "price", "costly"],
            "异味": ["味", "臭", "异味", "smell", "odor", "stink"],
        }
        for label, keywords in keyword_patterns.items():
            if label in self.labels:
                for kw in keywords:
                    self._keyword_map[kw.lower()] = label

    def annotate(self, text: str) -> AnnotationResult:
        """对单条文本进行 LLM 标注"""
        if self.simulate:
            return self._simulate_annotate(text)
        return self._call_llm(text)

    def annotate_batch(self, texts: list[str]) -> list[AnnotationResult]:
        """批量标注"""
        return [self.annotate(t) for t in texts]

    def _simulate_annotate(self, text: str) -> AnnotationResult:
        """模拟 LLM 标注（基于关键词规则 + 随机噪声）"""
        text_lower = text.lower()

        # 计算每个标签的匹配得分
        scores: dict[str, float] = {label: 0.0 for label in self.labels}
        for kw, label in self._keyword_map.items():
            if kw in text_lower:
                scores[label] += 1.0

        # 加入随机噪声（模拟 LLM 的不确定性）
        for label in scores:
            scores[label] += random.gauss(0, 0.3)

        # 选择最高分标签
        best_label = max(scores, key=lambda k: scores[k])

        # 模拟准确率：有一定概率标错
        if random.random() > self.simulate_accuracy:
            # 随机选择一个其他标签
            other_labels = [l for l in self.labels if l != best_label]
            if other_labels:
                best_label = random.choice(other_labels)

        # 根据得分差距确定置信度
        sorted_scores = sorted(scores.values(), reverse=True)
        margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 999

        if margin > 1.5:
            confidence = "high"
        elif margin > 0.5:
            confidence = "medium"
        else:
            confidence = "low"

        # 生成理由
        matched_kws = [kw for kw in self._keyword_map if kw in text_lower]
        reasoning = f"匹配关键词: {', '.join(matched_kws[:3])}" if matched_kws else "无明确关键词匹配"

        return AnnotationResult(
            text=text,
            label=best_label,
            confidence=confidence,
            reasoning=reasoning,
            model_name="simulated-llm",
        )

    def _call_llm(self, text: str) -> AnnotationResult:
        """真实 LLM API 调用（生产环境实现）

        特性:
            - 指数退避重试 (max_retries)
            - 标签合法性校验与模糊匹配
            - API 失败降级返回 unknown + low 置信度
        """
        prompt = self.prompt_template.replace("{{text}}", text)

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                raw_label = response.choices[0].message.content.strip()

                # ── 标签后处理 ──────────────────────────────
                label = self._normalize_label(raw_label)

                return AnnotationResult(
                    text=text,
                    label=label,
                    confidence="high",
                    reasoning=f"LLM({self.model}) 直接标注",
                    model_name=self.model,
                )

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "LLM API 调用失败 (尝试 %d/%d): %s",
                    attempt + 1,
                    self.max_retries,
                    exc,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避: 1s, 2s, 4s

        # 所有重试失败 → 降级返回
        logger.error("LLM API 最终失败: %s", last_error)
        return AnnotationResult(
            text=text,
            label="unknown",
            confidence="low",
            reasoning=f"API 调用失败: {last_error}",
            model_name=self.model,
        )

    def _normalize_label(self, raw: str) -> str:
        """将 LLM 原始输出归一化为合法标签"""
        raw_clean = raw.strip().strip('"').strip("'")

        # 精确匹配
        if raw_clean in self.labels:
            return raw_clean

        # 模糊匹配：大小写不敏感 + 子串匹配
        raw_lower = raw_clean.lower()
        for valid in self.labels:
            if valid.lower() == raw_lower:
                return valid
            if valid.lower() in raw_lower or raw_lower in valid.lower():
                return valid

        # 未匹配到已知标签
        logger.warning("LLM 返回未知标签 '%s'，回退到 'unknown'", raw_clean)
        return "unknown"


# ── 测试 ──────────────────────────────────────────────────────

def test_annotator():
    """测试 LLM 标注器"""
    print("=" * 60)
    print("测试: LLMAnnotator")
    print("=" * 60)

    labels = ["尺码偏差", "材质问题", "漏尿", "腰贴问题", "物流延迟", "过敏反应", "价格问题", "异味"]

    # 少样本示例
    few_shot = [
        {"text": "纸尿裤太大了，宝宝穿上松松垮垮", "label": "尺码偏差"},
        {"text": "这个面料太硬了，摩擦宝宝皮肤", "label": "材质问题"},
        {"text": "晚上总是漏尿，床单都湿了", "label": "漏尿"},
    ]

    annotator = LLMAnnotator(
        labels=labels,
        few_shot_examples=few_shot,
        simulate=True,
        simulate_accuracy=0.85,
    )

    test_texts = [
        "这个尺码偏小，建议买大一码",
        "腰贴粘不牢，宝宝一动就开了",
        "物流太慢了，等了两周才到",
        "宝宝用了以后皮肤发红，是不是过敏",
        "有一股很难闻的化学味道",
        "性价比还可以，就是包装破了",
    ]

    print("\n--- 单条标注 ---")
    for text in test_texts:
        result = annotator.annotate(text)
        print(f"\n文本: {result.text}")
        print(f"  标签: {result.label} (置信度: {result.confidence})")
        print(f"  理由: {result.reasoning}")

    print("\n--- 批量标注统计 ---")
    results = annotator.annotate_batch(test_texts)
    high_conf = sum(1 for r in results if r.confidence == "high")
    med_conf = sum(1 for r in results if r.confidence == "medium")
    low_conf = sum(1 for r in results if r.confidence == "low")
    print(f"高置信度: {high_conf}, 中置信度: {med_conf}, 低置信度: {low_conf}")

    print("\n" + "=" * 60)
    print("LLM 标注器测试完成 ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_annotator()
