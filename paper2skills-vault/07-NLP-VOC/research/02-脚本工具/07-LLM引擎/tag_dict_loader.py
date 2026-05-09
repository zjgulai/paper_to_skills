"""Tag Dictionary Loader - Phase 5 D2.T2.0

加载 v3.9 字典 (xlsx)，输出 LLM system prompt 友好的紧凑文本。

策略：
  - 9 个 sheet → 通用 + 6 品线 (吸奶器/内衣/家居/护理/喂养/智能)
  - 每个标签压成 1 行：TAG_ID|en|sentiment_mark|aipl
    例：TAG_L1_001|comfortable+|L1
  - 按 sheet 分组，便于 LLM 按品线检索
  - 提供 build_compact_prompt() 直接输出 system prompt 片段
  - 提供 ALL_TAG_IDS 集合用于 Pydantic 闭集校验
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

DICT_PATH = Path(
    "/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/01-字典版本/tag_dictionary_v3.9.xlsx"
)

TAG_SHEETS = [
    ("01_通用标签主表", "通用"),
    ("02_吸奶器", "吸奶器"),
    ("03_内衣服饰", "内衣服饰"),
    ("04_家居家纺", "家居家纺"),
    ("05_母婴综合护理", "母婴综合护理"),
    ("06_喂养电器", "喂养电器"),
    ("07_智能母婴电器", "智能母婴电器"),
]

SENTIMENT_MARK = {"正向": "+", "负向": "-", "中性": "·"}


@lru_cache(maxsize=1)
def load_dict_raw(path: str = str(DICT_PATH)) -> dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(path)
    return {name: pd.read_excel(xls, name) for name, _ in TAG_SHEETS if name in xls.sheet_names}


@lru_cache(maxsize=1)
def get_all_tag_ids(path: str = str(DICT_PATH)) -> frozenset[str]:
    sheets = load_dict_raw(path)
    ids: set[str] = set()
    for df in sheets.values():
        if "标签ID" in df.columns:
            ids.update(df["标签ID"].dropna().astype(str).tolist())
    return frozenset(ids)


def _format_tag_line(row: pd.Series) -> str:
    tid = str(row.get("标签ID", "")).strip()
    en = str(row.get("VOC标签（英文）", "")).strip().replace("|", "/").replace("\n", " ")
    if len(en) > 50:
        en = en[:48] + ".."
    sentiment = str(row.get("情感极性", "")).strip()
    mark = SENTIMENT_MARK.get(sentiment, "?")
    aipl = str(row.get("AIPL节点", "")).strip()
    return f"{tid}|{en}{mark}|{aipl}"


def build_compact_prompt(path: str = str(DICT_PATH), dedupe: bool = True) -> str:
    sheets = load_dict_raw(path)
    parts: list[str] = []
    parts.append("# 标签字典 (v3.9, 602 唯一标签)")
    parts.append("格式: TAG_ID|英文名{+正/-负/·中}|AIPL节点")
    parts.append("可用 AIPL 节点: A(认知) I(兴趣) P1(比价) P2(购买) L1(首用/产品) L2(售后/复购) L3(推荐/UGC)")
    parts.append("")
    seen: set[str] = set()
    for sheet_name, label in TAG_SHEETS:
        if sheet_name not in sheets:
            continue
        df = sheets[sheet_name]
        if "标签ID" not in df.columns:
            continue
        df = df[df["标签ID"].notna()]
        rows: list[tuple[str, str]] = []
        for _, row in df.iterrows():
            tid = str(row.get("标签ID", "")).strip()
            if dedupe and tid in seen:
                continue
            seen.add(tid)
            rows.append((tid, _format_tag_line(row)))
        if not rows:
            continue
        parts.append(f"## {label} ({len(rows)})")
        parts.extend(line for _, line in rows)
        parts.append("")
    return "\n".join(parts)


def write_compact_dict(out_path: Path, dict_path: str = str(DICT_PATH)):
    text = build_compact_prompt(dict_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return text


def stats(path: str = str(DICT_PATH)) -> dict:
    sheets = load_dict_raw(path)
    total = sum(len(df[df.get("标签ID").notna()]) for df in sheets.values() if "标签ID" in df.columns)
    by_sheet = {}
    for sheet_name, label in TAG_SHEETS:
        if sheet_name in sheets:
            df = sheets[sheet_name]
            if "标签ID" in df.columns:
                by_sheet[label] = int(df["标签ID"].notna().sum())
    return {"total_tags": total, "by_sheet": by_sheet, "all_tag_ids_count": len(get_all_tag_ids(path))}


if __name__ == "__main__":
    import argparse, sys

    ap = argparse.ArgumentParser()
    ap.add_argument("--write", type=Path, default=None, help="Write compact dict to file")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    if args.stats:
        s = stats()
        print(json.dumps(s, indent=2, ensure_ascii=False))
        sys.exit(0)

    text = build_compact_prompt()
    chars = len(text)
    approx_tokens = chars // 3
    lines = text.count("\n") + 1
    print(f"compact dict: {chars} chars / {lines} lines / ~{approx_tokens} tokens (1 token≈3 chars cn)")
    print(f"all_tag_ids count: {len(get_all_tag_ids())}")
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text, encoding="utf-8")
        print(f"written → {args.write}")
    else:
        print("\n--- preview (first 30 lines) ---")
        print("\n".join(text.split("\n")[:30]))
