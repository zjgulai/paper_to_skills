"""Phase 5 D11 T11.3.5 — BI Dashboard Spec Validator

校验 phase5-bi-dashboard-spec.md 是否覆盖 7 部门 × 5 章节 = 35 个断言，
且文档中无 TBD 关键词。

校验规则：
  - 每个部门必须有形如 `## <部门名>` 或 `### <部门名>` 的二/三级标题
  - 每个部门标题下，必须包含全部 required-sections 的子标题
  - 全文不得含 `TBD`/`tbd`/`todo`（大小写无关）
  - 失败时打印缺失项清单，exit 1；全过 exit 0
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


HEADING_RE = re.compile(r"^(#{2,4})\s+(.+?)\s*$", re.MULTILINE)
TBD_RE = re.compile(r"\b(TBD|todo)\b", re.IGNORECASE)


def parse_sections(md: str) -> list[tuple[int, str, int]]:
    out = []
    lines = md.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^(#{2,4})\s+(.+?)\s*$", line)
        if m:
            out.append((len(m.group(1)), m.group(2).strip(), i))
    return out


def find_dept_blocks(
    headings: list[tuple[int, str, int]], departments: list[str]
) -> dict[str, tuple[int, int]]:
    dept_set = set(departments)
    blocks: dict[str, tuple[int, int]] = {}
    for idx, (lvl, title, line_idx) in enumerate(headings):
        clean = title
        for sep in ("（", " ", "—", "-"):
            if sep in clean:
                clean = clean.split(sep)[0]
        clean = clean.strip()
        if clean in dept_set:
            end = len(headings)
            for j in range(idx + 1, len(headings)):
                if headings[j][0] <= lvl:
                    end = j
                    break
            blocks[clean] = (idx, end)
    return blocks


def validate(
    md_path: Path,
    departments: list[str],
    sections: list[str],
) -> tuple[bool, list[str]]:
    md = md_path.read_text(encoding="utf-8")
    failures: list[str] = []

    if TBD_RE.search(md):
        failures.append("文档含 TBD/todo 关键词（不允许）")

    headings = parse_sections(md)
    dept_blocks = find_dept_blocks(headings, departments)

    for dept in departments:
        if dept not in dept_blocks:
            failures.append(f"缺部门：{dept}")
            continue
        start, end = dept_blocks[dept]
        block_titles = {h[1] for h in headings[start + 1 : end]}

        for sec in sections:
            found = any(sec in t for t in block_titles)
            if not found:
                failures.append(f"{dept} 缺章节：{sec}")

    return (not failures), failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 5 D11 T11.3.5 BI Spec Validator")
    ap.add_argument("--spec", required=True, type=Path, help="BI spec markdown 路径")
    ap.add_argument(
        "--required-departments", required=True,
        help="逗号分隔的部门清单",
    )
    ap.add_argument(
        "--required-sections", required=True,
        help="逗号分隔的章节清单",
    )
    args = ap.parse_args(argv)

    if not args.spec.is_file():
        print(f"❌ spec 不存在: {args.spec}", file=sys.stderr); return 2

    departments = [d.strip() for d in args.required_departments.split(",") if d.strip()]
    sections = [s.strip() for s in args.required_sections.split(",") if s.strip()]
    expected_assertions = len(departments) * len(sections)

    print(f"⏳ 校验 spec: {args.spec}", file=sys.stderr)
    print(f"   部门 × 章节 = {len(departments)} × {len(sections)} = "
          f"{expected_assertions} 个断言 + TBD 检查", file=sys.stderr)

    ok, failures = validate(args.spec, departments, sections)
    if ok:
        print(f"✅ 35 个断言全过（{len(departments)} 部门 × {len(sections)} 章节）"
              "+ 无 TBD 关键词", file=sys.stderr)
        return 0
    print(f"❌ {len(failures)} 项不通过：", file=sys.stderr)
    for f in failures:
        print(f"   - {f}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
