"""ALCHEmist Label Function 生成（Phase 2.6）

为审核通过的候选标签生成可审计的 Python 标注规则。
每个 label function 包含：
- 关键词匹配（核心短语 + 语义变体）
- 否定词检测
- 排除词过滤
- 置信度评分
"""

import json
from pathlib import Path


NEGATION_WORDS = {
    "not", "no", "never", "none", "nobody", "nothing",
    "n't", "dont", "doesnt", "didnt", "wouldnt", "couldnt",
    "shouldnt", "wont", "cant", "isnt", "arent", "wasnt",
    "werent", "hasnt", "havent", "hadnt",
}


def generate_label_function(candidate: dict) -> str:
    """为单个候选标签生成 label function 代码"""
    tag_en = candidate["tag_en"]
    tag_cn = candidate["tag_cn"]
    aipl = candidate["aipl"]
    sentiment = candidate["sentiment"]
    category = candidate["category"]
    phrase = candidate["tag_en"].replace("_", " ")

    # 生成语义变体
    words = phrase.split()
    variants = [phrase]
    if len(words) >= 2:
        # 添加词序变体（如 "baby stroller" -> "stroller baby" 不太可能，但可以添加单数/复数）
        variants.append(" ".join(words))  # 原样
        # 单数/复数变体
        if words[-1].endswith("s"):
            variants.append(" ".join(words[:-1] + [words[-1][:-1]]))
        else:
            variants.append(" ".join(words[:-1] + [words[-1] + "s"]))

    # 去重
    variants = list(dict.fromkeys(variants))
    variant_str = ", ".join(f'"{v}"' for v in variants[:4])

    # 推断排除词（基于情感相反的情况）
    exclusion_rules = ""
    if sentiment == "positive":
        exclusion_rules = '''
    # 排除否定语境
    exclusion_phrases = ["not worth", "waste of", "disappointed with", "regret buying"]
    for ex in exclusion_phrases:
        if ex in text_lower:
            return False, 0.0'''
    elif sentiment == "negative":
        exclusion_rules = '''
    # 排除正面语境（可能是对比）
    if "but" in text_lower or "however" in text_lower:
        # 检查后半句是否转折为正面
        but_idx = text_lower.find("but")
        if but_idx > 0:
            after_but = text_lower[but_idx:]
            pos_words = ["good", "great", "love", "perfect", "excellent"]
            if any(w in after_but for w in pos_words):
                return False, 0.0'''

    code = f'''def lf_{tag_en}(text: str) -> tuple[bool, float]:
    """Label Function: {tag_cn} ({category}) -> {aipl}/{sentiment}

    触发条件: 文本包含以下关键词之一
    关键词: {variant_str}
    """
    text_lower = text.lower()

    # 核心关键词
    keywords = [{variant_str}]

    # 匹配关键词
    matched = False
    match_kw = ""
    for kw in keywords:
        if kw in text_lower:
            matched = True
            match_kw = kw
            break

    if not matched:
        return False, 0.0

    # 否定词检测
    idx = text_lower.find(match_kw)
    if idx >= 0:
        prefix = text_lower[max(0, idx - 20):idx]
        if any(neg in prefix for neg in {NEGATION_WORDS}):
            return False, 0.0
    {exclusion_rules}

    # 置信度
    confidence = min(0.4 + len(match_kw) * 0.05, 0.85)
    return True, round(confidence, 2)
'''
    return code


def generate_label_function_module(candidates: list[dict]) -> str:
    """生成完整的 label function 模块"""
    header = '''"""ALCHEmist Label Functions (Auto-Generated)

为标签字典 v3.4 新增候选标签生成的标注规则。
每个函数遵循: (text) -> (matched: bool, confidence: float)

Usage:
    from alchemist_label_functions import lf_bath_bombs, lf_wipe_warmer
    matched, conf = lf_bath_bombs("These bath bombs smell amazing!")
"""

NEGATION_WORDS = {
    "not", "no", "never", "none", "nobody", "nothing",
    "n't", "dont", "doesnt", "didnt", "wouldnt", "couldnt",
    "shouldnt", "wont", "cant", "isnt", "arent", "wasnt",
    "werent", "hasnt", "havent", "hadnt",
}


'''

    functions = []
    registry = []

    for c in candidates:
        func_code = generate_label_function(c)
        functions.append(func_code)
        registry.append(f'    "{c["tag_en"]}": lf_{c["tag_en"]},  # {c["tag_cn"]}')

    # 注册表
    registry_code = "\n".join(registry)
    footer = f'''
# ── 注册表 ────────────────────────────────────────────────────────

LABEL_FUNCTION_REGISTRY = {{
{registry_code}
}}


def apply_all(text: str) -> dict[str, tuple[bool, float]]:
    """对单条文本应用全部 label functions"""
    results = {{}}
    for tag_name, lf in LABEL_FUNCTION_REGISTRY.items():
        matched, conf = lf(text)
        if matched:
            results[tag_name] = (matched, conf)
    return results
'''

    return header + "\n\n".join(functions) + footer


def main():
    print("=" * 70)
    print("Phase 2.6: ALCHEmist Label Function 生成")
    print("=" * 70)

    base_dir = Path(__file__).parent.parent.parent / "04-输出结果"

    # 1. 加载自动通过的候选标签
    print("\n--- 加载候选标签 ---")
    approved_path = base_dir / "tag_gap_analysis/auto_approved_candidates.json"
    with open(approved_path, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    print(f"  候选标签: {len(candidates)} 个")

    # 2. 生成 label function 模块
    print("\n--- 生成 Label Functions ---")
    module_code = generate_label_function_module(candidates)

    # 3. 保存
    output_dir = Path(__file__).parent.parent.parent / "04-输出结果"
    output_dir.mkdir(parents=True, exist_ok=True)

    code_path = output_dir / "alchemist_label_functions.py"
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(module_code)
    print(f"  输出: {code_path}")

    # 统计
    n_lines = len(module_code.splitlines())
    n_funcs = len(candidates)
    print(f"  函数数: {n_funcs}")
    print(f"  代码行数: {n_lines}")

    # 4. 审计
    audit = {
        "phase": "2.6",
        "generated_functions": n_funcs,
        "code_lines": n_lines,
        "output_path": str(code_path),
    }
    audit_path = output_dir / "phase2_6_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"\n  审计: {audit_path}")

    print("\n" + "=" * 70)
    print("Phase 2.6 完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
