"""LLM 程序生成器

支持真实 LLM API 调用（OpenAI/DeepSeek/兼容接口）自动生成 Label Functions。
默认 model_name="simulated" 保持本地开发可用；设为其他值（如 "gpt-4o-mini"）
即可启用真实 API。
"""

from __future__ import annotations

import ast
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

from label_function import LabelFunction

# ── 可选依赖：openai SDK ──────────────────────────────────────
try:
    import openai

    _HAS_OPENAI = True
except ImportError:
    openai = None  # type: ignore
    _HAS_OPENAI = False

logger = logging.getLogger(__name__)

# ── 安全配置：exec 可用的安全命名空间 ──────────────────────────
_safe_builtins = {
    "any": any,
    "all": all,
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "range": range,
    "enumerate": enumerate,
    "zip": zip,
    "max": max,
    "min": min,
    "sum": sum,
    "abs": abs,
    "round": round,
    "print": print,
    "re": re,
}


class ProgramGenerator:
    """标注程序生成器

    模式切换:
        model_name="simulated" (默认) → 本地关键词提取生成
        model_name="gpt-4o-mini"      → 调用真实 LLM API 生成 Python 函数

    生产环境配置（环境变量）:
        OPENAI_API_KEY    : API 密钥
        OPENAI_BASE_URL   : 自定义 API 基地址（如 DeepSeek）
        OPENAI_MODEL      : 模型名称，默认 gpt-4o-mini
    """

    def __init__(
        self,
        model_name: str = "simulated",
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_retries: int = 3,
        timeout: float = 60.0,
        max_tokens: int = 2048,
    ):
        self.model_name = model_name
        self.simulate = model_name == "simulated"

        # API 配置（仅非模拟模式有效）
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_tokens = max_tokens

        # 初始化真实 API 客户端
        self._client: Optional[object] = None
        if not self.simulate:
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
            logger.info(
                "ProgramGenerator 已接入模型: %s (model_name=%s)",
                self.model,
                self.model_name,
            )

    def generate(
        self,
        label_name: str,
        description: str,
        positive_examples: list[str],
        negative_examples: Optional[list[str]] = None,
        n_programs: int = 3,
    ) -> list[LabelFunction]:
        """为给定标签生成多个标注程序

        Args:
            label_name: 标签名称（如"过敏反应"）
            description: 标签描述
            positive_examples: 正例文本列表
            negative_examples: 反例文本列表
            n_programs: 生成程序数量

        Returns:
            生成的 LabelFunction 列表
        """
        if self.simulate:
            return self._generate_simulated(
                label_name, description, positive_examples, negative_examples, n_programs
            )
        return self._generate_with_llm(
            label_name, description, positive_examples, negative_examples, n_programs
        )

    def _generate_simulated(
        self,
        label_name: str,
        description: str,
        positive_examples: list[str],
        negative_examples: Optional[list[str]] = None,
        n_programs: int = 3,
    ) -> list[LabelFunction]:
        """模拟 LLM 程序生成

        基于正例提取关键词，生成不同策略的程序：
        - 策略1: 关键词匹配
        - 策略2: 正则表达式
        - 策略3: 组合条件
        """
        neg_examples = negative_examples or []

        # 从正例中提取高频词（简单分词）
        all_words: dict[str, int] = {}
        for text in positive_examples:
            for i in range(len(text) - 1):
                word = text[i : i + 2]
                if len(word) == 2 and any("\u4e00" <= c <= "\u9fff" for c in word):
                    all_words[word] = all_words.get(word, 0) + 1

        # 过滤在负例中出现的高频词
        neg_words = set()
        for text in neg_examples:
            for i in range(len(text) - 1):
                neg_words.add(text[i : i + 2])

        keywords = [
            w
            for w, c in sorted(all_words.items(), key=lambda x: x[1], reverse=True)
            if w not in neg_words and c >= 2
        ][:8]

        programs: list[LabelFunction] = []

        # 策略1: 关键词匹配
        if keywords and n_programs >= 1:
            kw_list = keywords[:5]
            programs.append(
                LabelFunction(
                    name=f"lf_{label_name}_keywords",
                    func=lambda t, kws=kw_list, ln=label_name: ln
                    if any(kw in t for kw in kws)
                    else None,
                    description=f"关键词匹配: {', '.join(kw_list)}",
                    source="llm_generated",
                )
            )

        # 策略2: 正则模式匹配（模拟）
        if n_programs >= 2 and keywords:
            patterns = keywords[:3]
            programs.append(
                LabelFunction(
                    name=f"lf_{label_name}_pattern",
                    func=lambda t, pts=patterns, ln=label_name: ln
                    if any(p in t for p in pts)
                    else None,
                    description=f"模式匹配: {', '.join(patterns)}",
                    source="llm_generated",
                )
            )

        # 策略3: 否定规则（排除反例特征）
        if n_programs >= 3 and neg_examples:
            neg_keywords = []
            for text in neg_examples[:3]:
                for i in range(len(text) - 1):
                    word = text[i : i + 2]
                    if len(word) == 2 and any("\u4e00" <= c <= "\u9fff" for c in word):
                        neg_keywords.append(word)
            neg_keywords = list(set(neg_keywords))[:3]

            if neg_keywords and keywords:
                programs.append(
                    LabelFunction(
                        name=f"lf_{label_name}_filtered",
                        func=lambda t, kws=keywords[:3], negs=neg_keywords, ln=label_name: (
                            ln
                            if any(kw in t for kw in kws)
                            and not any(n in t for n in negs)
                            else None
                        ),
                        description=f"过滤规则: 包含{', '.join(keywords[:3])} 但不包含{', '.join(neg_keywords)}",
                        source="llm_generated",
                    )
                )

        return programs

    def _generate_with_llm(
        self,
        label_name: str,
        description: str,
        positive_examples: list[str],
        negative_examples: Optional[list[str]] = None,
        n_programs: int = 3,
    ) -> list[LabelFunction]:
        """真实 LLM 程序生成（生产环境实现）

        流程:
            1. 构造 prompt 要求 LLM 输出 Python 函数
            2. 调用 API（带重试 + 指数退避）
            3. 提取 ```python ... ``` 代码块
            4. AST 安全检查 → exec 编译
            5. 包装为 LabelFunction 列表
            6. 失败时自动回退到 _generate_simulated
        """
        if not _HAS_OPENAI:
            logger.error("openai SDK 未安装，回退到模拟模式")
            return self._generate_simulated(
                label_name, description, positive_examples, negative_examples, n_programs
            )

        neg_text = ""
        if negative_examples:
            neg_text = "\n反例文本:\n" + "\n".join(
                f'- "{t}"' for t in negative_examples[:5]
            )

        prompt = f"""你是一名数据标注专家。请为以下标签编写 Python 函数，用于自动判断文本是否属于该标签。

标签名称: {label_name}
标签定义: {description}

正例文本:
{chr(10).join(f'- "{t}"' for t in positive_examples[:5])}
{neg_text}

要求:
1. 函数签名: def lf_{label_name.lower().replace(' ', '_')}(text: str) -> str | None:
2. 匹配时返回标签名称，不匹配时返回 None（弃权）
3. 使用简单的关键词匹配或正则表达式
4. 函数要轻量化，避免复杂逻辑
5. 请输出 {n_programs} 个不同的函数，每个函数用 ```python 和 ``` 包裹

请直接输出 Python 函数代码，不要解释。"""

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content
                programs = self._extract_functions(content, label_name)

                if programs:
                    logger.info(
                        "LLM 成功生成 %d 个程序 (label=%s)", len(programs), label_name
                    )
                    return programs
                else:
                    logger.warning(
                        "LLM 未返回有效函数 (label=%s)，尝试重试...", label_name
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
                    time.sleep(2 ** attempt)

        # 所有尝试失败 → 回退模拟模式
        logger.error(
            "LLM 程序生成最终失败 (label=%s): %s，回退到模拟模式",
            label_name,
            last_error,
        )
        return self._generate_simulated(
            label_name, description, positive_examples, negative_examples, n_programs
        )

    def _extract_functions(self, content: str, label_name: str) -> list[LabelFunction]:
        """从 LLM 响应中提取 Python 函数并编译为 LabelFunction"""
        # 提取 ```python ... ``` 代码块
        code_blocks = re.findall(r"```python\s*\n(.*?)```", content, re.DOTALL)
        if not code_blocks:
            # 尝试不带 python 标记的代码块
            code_blocks = re.findall(r"```\s*\n(.*?)```", content, re.DOTALL)

        programs: list[LabelFunction] = []
        for i, code in enumerate(code_blocks):
            code = code.strip()
            if not code:
                continue

            # 安全检查
            if not self._is_safe_code(code):
                logger.warning("代码块 %d 未通过安全检查，已跳过", i + 1)
                continue

            try:
                namespace: dict = {"__builtins__": _safe_builtins}
                exec(compile(code, "<llm_generated>", "exec"), namespace)

                # 收集生成的函数对象
                for name, obj in namespace.items():
                    if callable(obj) and name.startswith("lf_"):
                        programs.append(
                            LabelFunction(
                                name=f"{name}_v{i + 1}",
                                func=obj,
                                description=f"LLM({self.model}) 生成: {name}",
                                source="llm_generated",
                            )
                        )
                        if len(programs) >= 5:  # 上限保护
                            break

            except Exception as exc:
                logger.warning("代码块 %d 编译/执行失败: %s", i + 1, exc)
                continue

        return programs

    @staticmethod
    def _is_safe_code(code: str) -> bool:
        """AST 安全检查：禁止危险操作

        禁止:
            - import / from ... import
            - __ 双下划线属性访问
            - 非白名单的外部调用
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False

        for node in ast.walk(tree):
            # 禁止 import
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return False

            # 禁止 __ 双下划线属性
            if isinstance(node, ast.Attribute):
                if node.attr.startswith("__"):
                    return False

            # 禁止非白名单的函数调用（除了 re.xxx）
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id not in _safe_builtins:
                        return False
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        if node.func.value.id not in ("re", "str"):
                            return False
                    else:
                        return False

        return True


# ── 测试 ──────────────────────────────────────────────────────


def test_program_generator():
    """测试程序生成器"""
    print("=" * 60)
    print("测试: ProgramGenerator")
    print("=" * 60)

    generator = ProgramGenerator(model_name="simulated")

    # 为"过敏反应"标签生成程序
    label_name = "过敏反应"
    description = "用户反馈使用后出现皮肤过敏症状（红疹、发红、瘙痒等）"

    positive_examples = [
        "宝宝用了后起红疹，怀疑是过敏",
        "腰部一圈红红的，是不是过敏了",
        "大腿内侧发红，应该是过敏",
        "红屁屁严重，怀疑是过敏反应",
        "用了三天，屁股上起了小疙瘩",
        "之前用别的牌子没事，这个一用就红",
        "皮肤敏感的宝宝慎买",
        "材质可能不适合敏感肌",
    ]

    negative_examples = [
        "这个尺码偏小，勒得宝宝不舒服",
        "面料太硬了，摩擦得皮肤不舒服",
        "物流太慢了",
        "价格有点贵",
    ]

    print(f"\n--- 为标签 '{label_name}' 生成程序 ---")
    print(f"正例: {len(positive_examples)} 条")
    print(f"反例: {len(negative_examples)} 条")

    programs = generator.generate(
        label_name=label_name,
        description=description,
        positive_examples=positive_examples,
        negative_examples=negative_examples,
        n_programs=3,
    )

    print(f"\n生成 {len(programs)} 个程序:")
    for i, prog in enumerate(programs, 1):
        print(f"\n  [程序{i}] {prog.name}")
        print(f"  描述: {prog.description}")

    # 测试生成的程序
    test_texts = [
        "宝宝用了后起红疹",
        "腰部一圈红红的",
        "尺码偏小，勒得不舒服",
        "物流太慢了",
    ]

    print("\n--- 程序标注测试 ---")
    for text in test_texts:
        print(f"\n  文本: '{text}'")
        for prog in programs:
            result = prog(text)
            status = result if result else "[弃权]"
            print(f"    {prog.name}: {status}")

    print("\n" + "=" * 60)
    print("程序生成器测试完成 ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_program_generator()
