"""Stratified Sampler - Phase 5 D1.T1.4

5000 条分层抽样器，从 phase4_labeled.jsonl 抽取测试集。

分层维度（4 级）：
  1. data_source: amazon_competitor / trustpilot / zendesk / momcozy / reddit
  2. language: en / de / fr / zh / other
  3. length_bucket: short(<=50) / medium(<=200) / long(<=500) / xlong(>500)
  4. has_phase4_label: True / False

目标比例（D1 计划锁定）：
  Amazon 2670 / Trustpilot 1369 / Zendesk 647 / Momcozy 270 / Reddit 44 = 5000

策略：
  - 一级按 data_source 锁定目标 n_h
  - 二/三/四级按各源内部自然比例分配
  - 每个最小分层 ≥ 1 条（保证统计代表性）
  - 不足时按四级（最稀疏）的优先级回退
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INPUT = Path(
    "/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/04-输出结果/unified_labeling/phase4_labeled.jsonl"
)
DEFAULT_OUTPUT = Path(
    "/Users/pray/project/paper_to_skills/paper2skills-vault/07-NLP-VOC/research/03-数据资产/test_set_5k_stratified.jsonl"
)

TARGET_BY_SOURCE = {
    "amazon_competitor": 2670,
    "trustpilot": 1369,
    "zendesk": 647,
    "momcozy": 270,
    "reddit": 44,
}

LEN_BUCKETS = [
    ("short", 0, 50),
    ("medium", 51, 200),
    ("long", 201, 500),
    ("xlong", 501, 10**9),
]


def length_bucket(n: int) -> str:
    for name, lo, hi in LEN_BUCKETS:
        if lo <= n <= hi:
            return name
    return "xlong"


def normalize_lang(lang: str | None) -> str:
    if not lang:
        return "other"
    s = str(lang).lower().strip()
    if s in ("en", "de", "fr", "zh"):
        return s
    return "other"


def load_records(path: Path, max_lines: int | None = None) -> list[dict]:
    recs: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            recs.append(r)
            if max_lines and len(recs) >= max_lines:
                break
    return recs


def enrich(records: list[dict]) -> list[dict]:
    for r in records:
        text = r.get("text") or ""
        r["_text_len"] = len(text)
        r["_length_bucket"] = length_bucket(r["_text_len"])
        r["_lang_norm"] = normalize_lang(r.get("language"))
        r["_has_label"] = bool(r.get("labels")) and len(r.get("labels", [])) > 0
    return records


def stratified_allocate(
    records: list[dict],
    targets: dict[str, int],
    seed: int = 42,
) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(records)
    df = df[df["data_source"].isin(targets.keys())].copy()

    sampled_indices: list[int] = []
    audit: dict = {"by_source": {}, "by_strata": {}}

    for src, target_n in targets.items():
        src_df = df[df["data_source"] == src]
        if len(src_df) == 0:
            audit["by_source"][src] = {
                "available": 0,
                "target": target_n,
                "actual": 0,
                "warning": "source has no records",
            }
            continue

        if len(src_df) <= target_n:
            picked = src_df.index.tolist()
            sampled_indices.extend(picked)
            audit["by_source"][src] = {
                "available": len(src_df),
                "target": target_n,
                "actual": len(picked),
                "warning": "source size <= target, took all",
            }
            continue

        strata_groups = src_df.groupby(["_lang_norm", "_length_bucket", "_has_label"])
        strata_sizes = strata_groups.size()
        strata_total = strata_sizes.sum()

        per_stratum_target = (strata_sizes / strata_total * target_n).round().astype(int)
        per_stratum_target = per_stratum_target.clip(lower=1)

        diff = target_n - per_stratum_target.sum()
        if diff != 0:
            sorted_keys = strata_sizes.sort_values(ascending=False).index.tolist()
            i = 0
            while diff != 0 and sorted_keys:
                k = sorted_keys[i % len(sorted_keys)]
                if diff > 0:
                    per_stratum_target.loc[k] += 1
                    diff -= 1
                else:
                    if per_stratum_target.loc[k] > 1:
                        per_stratum_target.loc[k] -= 1
                        diff += 1
                i += 1

        src_actual = 0
        src_strata_audit = {}
        for stratum_key, take_n in per_stratum_target.items():
            stratum_df = strata_groups.get_group(stratum_key)
            n_take = min(take_n, len(stratum_df))
            picked = rng.choice(stratum_df.index.values, size=n_take, replace=False)
            sampled_indices.extend(picked.tolist())
            src_actual += n_take
            src_strata_audit[str(stratum_key)] = {
                "available": len(stratum_df),
                "target": int(take_n),
                "actual": int(n_take),
            }

        audit["by_source"][src] = {
            "available": len(src_df),
            "target": target_n,
            "actual": src_actual,
        }
        audit["by_strata"][src] = src_strata_audit

    sampled = df.loc[sampled_indices].sample(frac=1, random_state=seed).reset_index(drop=True)
    audit["total_sampled"] = len(sampled)
    audit["target_total"] = sum(targets.values())

    return sampled.to_dict("records"), audit


def write_jsonl(records: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            r_clean = {k: v for k, v in r.items() if not k.startswith("_")}
            f.write(json.dumps(r_clean, ensure_ascii=False) + "\n")


def print_audit(audit: dict):
    print("\n" + "=" * 70)
    print("Stratified Sampling Audit")
    print("=" * 70)
    print(f"Target total: {audit.get('target_total')}")
    print(f"Sampled total: {audit.get('total_sampled')}")
    print("\n按数据源:")
    print(f"{'data_source':25} {'available':>10} {'target':>8} {'actual':>8} {'note':30}")
    for src, info in audit["by_source"].items():
        warn = info.get("warning", "")
        print(f"  {src:25} {info['available']:>10} {info['target']:>8} {info['actual']:>8} {warn:30}")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser(description="Phase 5 stratified sampler")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--total", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--audit-out", type=Path, default=None, help="Audit JSON path")
    ap.add_argument("--max-input-lines", type=int, default=None, help="For debug only")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"❌ Input not found: {args.input}")
        return 2

    print(f"Loading: {args.input}")
    records = load_records(args.input, args.max_input_lines)
    print(f"Loaded {len(records)} records")

    enrich(records)

    if args.total != 5000:
        scale = args.total / 5000
        targets = {k: max(1, int(round(v * scale))) for k, v in TARGET_BY_SOURCE.items()}
        diff = args.total - sum(targets.values())
        if diff != 0:
            largest_src = max(targets, key=targets.get)
            targets[largest_src] += diff
    else:
        targets = TARGET_BY_SOURCE.copy()

    print(f"Targets per source: {targets}")
    print(f"Target sum: {sum(targets.values())}")

    sampled, audit = stratified_allocate(records, targets, seed=args.seed)

    write_jsonl(sampled, args.output)
    print(f"\nWrote {len(sampled)} records → {args.output}")

    print_audit(audit)

    audit_path = args.audit_out or args.output.parent / (args.output.stem + "_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Audit saved: {audit_path}")

    print("\n— 校验 —")
    pass_all = True
    df = pd.DataFrame(sampled)
    if "data_source" in df.columns:
        actual_counts = df["data_source"].value_counts().to_dict()
        for src, target_n in targets.items():
            actual = actual_counts.get(src, 0)
            tol = max(2, int(target_n * 0.02))
            ok = abs(actual - target_n) <= tol
            mark = "✓" if ok else "✗"
            print(f"  {mark} {src:25} actual={actual} target={target_n} tol=±{tol}")
            if not ok:
                pass_all = False

    if pass_all:
        print("\n🎉 D1.QA2 PASS — 抽样比例命中目标 ±2%")
        return 0
    else:
        print("\n❌ D1.QA2 FAIL — 比例偏离过大")
        return 1


if __name__ == "__main__":
    sys.exit(main())
