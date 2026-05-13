"""Apply L5 dept-rename to YAML files in-place.

Two safe constraints:
  1. Idempotent · re-running yields no further change.
  2. Q3=A · compound `部门：描述` strings rename only the prefix.

Mapping is the canonical 28+1 set ratified through Q1-Q7.
"""

from __future__ import annotations
import sys
import re
from pathlib import Path
from typing import Iterable

MAPPING_ORDERED = [
    ("全球客服与体验中心", "全球客服中心"),
    ("产品中心/品线",      "产品中心"),
    ("供应链中心",         "仓储物流部"),
    ("品控部",             "品质管理中心"),
    ("质量与法规部",       "法务合规部"),
    ("市场部",             "品牌市场中心"),
    ("产品市场部",         "产品中心"),
    ("客服部",             "全球客服中心"),
    ("会员运营部",         "用户运营部"),
    ("品牌运营部",         "品牌市场中心"),
    ("私域运营部",         "用户运营部"),
    ("产品研发部",         "产品中心"),
    ("声学实验室",         "产品中心"),
    ("国际物流部",         "仓储物流部"),
    ("产品规划部",         "产品中心"),
    ("物流运营部",         "仓储物流部"),
    ("产品运营部",         "产品中心"),
    ("产品部",             "产品中心"),
    ("全球营销服务中心",   "品牌市场中心"),
    ("营运部",             "电商运营部"),
    ("产品设计部",         "产品中心"),
    ("供应链部",           "仓储物流部"),
    ("研发部",             "产品中心"),
    ("设计部",             "产品中心"),
    ("公关",               "品牌市场中心"),
    ("法务",               "法务合规部"),
    ("用户体验部",         "全球客服中心"),
    ("产品品牌市场中心",   "品牌市场中心"),
    ("运营部",             "电商运营部"),
]

PROTECTED_RUNYING = (
    "电商运营部", "物流运营部", "品牌运营部",
    "私域运营部", "用户运营部", "会员运营部", "产品运营部",
)


def replace_with_protection(text: str) -> tuple[str, list[tuple[str, str, int]]]:
    """Return new text + list of (old, new, count) actually replaced."""
    out = text
    log: list[tuple[str, str, int]] = []
    for old, new in MAPPING_ORDERED:
        if old == "运营部":
            placeholders: list[tuple[str, str]] = []
            for i, p in enumerate(PROTECTED_RUNYING):
                ph = f"\x00P{i}\x00"
                if p in out:
                    out = out.replace(p, ph)
                    placeholders.append((ph, p))
            cnt = out.count(old)
            if cnt:
                out = out.replace(old, new)
                log.append((old, new, cnt))
            for ph, original in placeholders:
                out = out.replace(ph, original)
        else:
            cnt = out.count(old)
            if cnt:
                out = out.replace(old, new)
                log.append((old, new, cnt))
    return out, log


def process(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    new_text, log = replace_with_protection(text)
    if new_text == text:
        print(f"  · {path.name}: no change")
        return 0
    path.write_text(new_text, encoding="utf-8")
    total = sum(c for _, _, c in log)
    print(f"  · {path.name}: {total} replacements")
    for old, new, cnt in log:
        print(f"      {old:>20} → {new:<14} × {cnt}")
    return total


def main(argv: Iterable[str]) -> int:
    paths = [Path(a) for a in argv]
    if not paths:
        print("Usage: rename_dept_l5_yaml.py <file.yaml> [file.yaml ...]")
        return 2
    grand = 0
    for p in paths:
        if not p.is_file():
            print(f"  ! skip (not a file): {p}", file=sys.stderr)
            continue
        grand += process(p)
    print(f"\nTotal replacements across {len(paths)} files: {grand}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
