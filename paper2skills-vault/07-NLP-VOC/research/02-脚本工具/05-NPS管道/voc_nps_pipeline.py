from __future__ import annotations

"""VOC Proxy NPS Pipeline — 集成 NPS 校准层的完整工作流

工作流：
  1. 加载已分类 VOC 数据（含 product_line, category_lv4, rating, source）
  2. 按 (source × product_line × 可选 AIPL 节点) 分组计算原始 Proxy NPS
  3. Layer 2: NPSCalibrator 自动校准 source bias
  4. Layer 3: CategoryStandardizer 计算品类相对 NPS
  5. 输出三层指标 + 自证审计 + BI 可用 CSV
"""

import json
import sys
from pathlib import Path

import pandas as pd

# 引入校准引擎
sys.path.insert(0, str(Path(__file__).parent))
from nps_calibration import (
    DEFAULT_SOURCE_BASELINES,
    GLOBAL_BASELINE_SOURCE,
    NPSAuditor,
    NPSCalibrator,
    NPSDashboard,
    CategoryStandardizer,
)


# ─────────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────────

# 星级 → NPS 类别映射
RATING_TO_NPS_CLASS = {
    1: "detractor",
    2: "detractor",
    3: "neutral",
    4: "promoter",
    5: "promoter",
}

# 默认分组维度
DEFAULT_GROUP_COLS = ["data_source", "product_line"]


# ─────────────────────────────────────────────────────────────────
# 核心 Pipeline
# ─────────────────────────────────────────────────────────────────

class VOCNPSPipeline:
    """VOC NPS Pipeline：从原始数据到三层校准指标"""

    def __init__(
        self,
        calibrator: NPSCalibrator | None = None,
        standardizer: CategoryStandardizer | None = None,
        group_cols: list[str] | None = None,
    ):
        self.calibrator = calibrator or NPSCalibrator()
        self.standardizer = standardizer
        self.group_cols = group_cols or DEFAULT_GROUP_COLS.copy()
        self.audit_report: dict | None = None

    def _resolve_source(self, source: str) -> str:
        """将变体 data_source 映射到标准源名（如 momcozy_amazon → amazon）"""
        source = source.lower().strip()
        known = set(self.calibrator.source_bias.keys())
        if source in known:
            return source
        # 子串匹配
        for k in known:
            if k in source or source in k:
                return k
        return source

    # ── Step 1: 加载数据 ─────────────────────────────────────────

    def load(self, csv_path: Path) -> pd.DataFrame:
        """加载已分类 VOC 数据"""
        print(f"[Pipeline] 加载数据: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"  总记录: {len(df):,}")

        # 确保必要列存在
        required = {"rating", "data_source"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"数据缺少必要列: {missing}")

        # 清洗 rating 为整数
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df = df.dropna(subset=["rating"])
        df["rating"] = df["rating"].astype(int)

        # 映射 NPS 类别
        df["nps_class"] = df["rating"].map(RATING_TO_NPS_CLASS)

        # 清洗 data_source
        df["data_source"] = df["data_source"].str.lower().str.strip()

        print(f"  有效记录: {len(df):,}")
        return df

    # ── Step 2: 计算原始 Proxy NPS ───────────────────────────────

    def compute_raw_nps(self, df: pd.DataFrame) -> pd.DataFrame:
        """按分组维度计算原始 NPS"""
        print(f"\n[Pipeline] 按 {self.group_cols} 分组计算原始 NPS...")

        # 过滤掉缺失分组维度的记录
        for col in self.group_cols:
            if col not in df.columns:
                raise ValueError(f"分组列 '{col}' 不在数据中")
            df = df[df[col].notna()]

        results = []
        for keys, sub in df.groupby(self.group_cols):
            keys = keys if isinstance(keys, tuple) else (keys,)
            row = dict(zip(self.group_cols, keys))

            total = len(sub)
            promoter = (sub["nps_class"] == "promoter").sum()
            detractor = (sub["nps_class"] == "detractor").sum()
            neutral = (sub["nps_class"] == "neutral").sum()

            raw_nps = (promoter - detractor) / total * 100 if total > 0 else 0.0

            row.update({
                "total_records": total,
                "promoter_count": int(promoter),
                "detractor_count": int(detractor),
                "neutral_count": int(neutral),
                "promoter_pct": round(promoter / total * 100, 2) if total > 0 else 0.0,
                "detractor_pct": round(detractor / total * 100, 2) if total > 0 else 0.0,
                "raw_nps": round(raw_nps, 2),
            })
            results.append(row)

        result_df = pd.DataFrame(results)
        print(f"  共 {len(result_df)} 个分组")
        return result_df

    # ── Step 3: Layer 2 — 数据源校准 ─────────────────────────────

    def apply_source_calibration(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用数据源偏差校准"""
        print("\n[Pipeline] Layer 2: 数据源偏差校准...")

        def _calibrate_row(row: pd.Series) -> pd.Series:
            source = self._resolve_source(str(row["data_source"]))
            raw = float(row["raw_nps"])
            try:
                r = self.calibrator.calibrate(source, raw)
                return pd.Series({
                    "source_bias": round(r.source_bias, 2),
                    "layer2_calibrated_nps": round(r.calibrated_nps, 2),
                })
            except ValueError:
                import warnings
                warnings.warn(f"[NPSPipeline] 未知数据源 '{source}'，跳过 source bias 校准，使用原始 NPS={raw}")
                return pd.Series({
                    "source_bias": None,
                    "layer2_calibrated_nps": raw,
                })

        calib_cols = df.apply(_calibrate_row, axis=1)
        df = pd.concat([df, calib_cols], axis=1)

        calibrated_count = df["source_bias"].notna().sum()
        print(f"  已校准 {calibrated_count} / {len(df)} 个分组")
        return df

    # ── Step 4: Layer 3 — 品类标准化 ─────────────────────────────

    def apply_category_standardization(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用品类相对标准化（需要 category_lv4 列）"""
        if self.standardizer is None:
            print("\n[Pipeline] Layer 3: 跳过（无品类基线）")
            df["category_bias"] = 0.0
            df["layer3_category_relative_nps"] = df["layer2_calibrated_nps"]
            return df

        print("\n[Pipeline] Layer 3: 品类相对标准化...")

        # 使用 product_line 作为品类代理（若 category_lv4 不在数据中）
        category_col = "category_lv4" if "category_lv4" in df.columns else "product_line"

        def _standardize_row(row: pd.Series) -> pd.Series:
            cat = str(row.get(category_col, ""))
            calibrated = float(row["layer2_calibrated_nps"])
            # 品类偏差 = 该品类的平均 NPS（作为基线）
            cat_avg = self.standardizer.category_stats.get(cat, {}).get("avg_nps", 0.0)
            if cat not in self.standardizer.category_stats:
                return pd.Series({"category_bias": 0.0, "layer3_category_relative_nps": calibrated})
            return pd.Series({
                "category_bias": round(cat_avg, 2),
                "layer3_category_relative_nps": round(calibrated - cat_avg, 2),
            })

        std_cols = df.apply(_standardize_row, axis=1)
        df = pd.concat([df, std_cols], axis=1)

        applied = (df["category_bias"] != 0).sum()
        print(f"  已标准化 {applied} / {len(df)} 个分组")
        return df

    # ── Step 5: 自证审计 ─────────────────────────────────────────

    def run_audit(self, df: pd.DataFrame) -> dict:
        """对校准结果执行自证审计"""
        print("\n[Pipeline] 自证审计...")

        # 构造 CalibratedResult 列表用于审计
        from nps_calibration import CalibratedResult

        results = []
        for _, row in df.iterrows():
            if pd.notna(row.get("source_bias")):
                results.append(CalibratedResult(
                    source_name=str(row["data_source"]),
                    raw_nps=float(row["raw_nps"]),
                    calibrated_nps=float(row["layer2_calibrated_nps"]),
                    source_bias=float(row["source_bias"]),
                ))

        auditor = NPSAuditor(self.calibrator)
        self.audit_report = auditor.audit(results)

        print(f"  状态: {self.audit_report['status']}")
        print(f"  问题: {self.audit_report['issue_count']} 个")
        print(f"  警告: {self.audit_report['warning_count']} 个")
        for issue in self.audit_report["issues"]:
            print(f"    ❌ {issue}")
        for warn in self.audit_report["warnings"]:
            print(f"    ⚠️ {warn}")

        return self.audit_report

    # ── Step 6: 输出 ─────────────────────────────────────────────

    def export(
        self,
        df: pd.DataFrame,
        output_dir: Path,
        tag: str = "voc_nps",
    ) -> dict:
        """导出三层指标到 BI 可用格式"""
        print(f"\n[Pipeline] 导出到 {output_dir}...")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 完整三层指标 CSV（BI 用）
        csv_path = output_dir / f"{tag}_3layer_dashboard.csv"

        # 添加样本量警告标记
        df["sample_size_warning"] = df["total_records"].apply(
            lambda n: "LOW" if n < 30 else ("CAUTION" if n < 100 else "OK")
        )

        # 确定输出列顺序
        core_cols = self.group_cols + [
            "total_records",
            "sample_size_warning",
            "promoter_count",
            "detractor_count",
            "neutral_count",
            "promoter_pct",
            "detractor_pct",
            "raw_nps",
            "source_bias",
            "layer2_calibrated_nps",
            "category_bias",
            "layer3_category_relative_nps",
        ]
        # 只保留实际存在的列
        output_cols = [c for c in core_cols if c in df.columns]
        df_export = df[output_cols].copy()
        df_export.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  三层看板 CSV: {csv_path}")

        # 2. JSON 报告（程序化消费）
        report = {
            "meta": {
                "pipeline_version": "1.0.0",
                "group_dimensions": self.group_cols,
                "total_groups": len(df),
                "global_baseline": self.calibrator.global_source,
                "global_nps": self.calibrator.global_baseline.nps,
            },
            "audit": self.audit_report,
            "summary": {
                "by_source": {},
            },
            "groups": df_export.to_dict("records"),
        }

        # 按 source 汇总
        for source in df["data_source"].unique():
            sub = df[df["data_source"] == source]
            report["summary"]["by_source"][source] = {
                "group_count": len(sub),
                "total_records": int(sub["total_records"].sum()),
                "avg_raw_nps": round(sub["raw_nps"].mean(), 2),
                "avg_calibrated_nps": round(
                    sub["layer2_calibrated_nps"].mean(), 2
                ) if "layer2_calibrated_nps" in sub.columns else None,
            }

        json_path = output_dir / f"{tag}_pipeline_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"  JSON 报告: {json_path}")

        # 3. Markdown 摘要
        md_path = output_dir / f"{tag}_summary.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# VOC NPS Pipeline 结果摘要\n\n")
            f.write(f"**分组维度**: {' × '.join(self.group_cols)}\n\n")
            f.write(f"**总分组数**: {len(df)}\n\n")
            f.write(f"**全局基准**: {self.calibrator.global_source}\n\n")
            f.write(f"**审计状态**: {self.audit_report['status'] if self.audit_report else 'N/A'}\n\n")

            # 按 source 的对比表
            f.write("## 数据源对比（原始 vs 校准后）\n\n")
            f.write("| 数据源 | 分组数 | 总记录 | 平均原始 NPS | 平均校准后 NPS |\n")
            f.write("|--------|--------|--------|-------------|---------------|\n")
            for source, info in report["summary"]["by_source"].items():
                cal = info.get("avg_calibrated_nps")
                cal_str = f"{cal:.1f}" if cal is not None else "N/A"
                f.write(f"| {source} | {info['group_count']} | {info['total_records']:,} | "
                        f"{info['avg_raw_nps']:.1f} | {cal_str} |\n")

            # Top/Bottom product_line
            if "product_line" in df.columns and "layer2_calibrated_nps" in df.columns:
                f.write("\n## 品线 NPS Top 10\n\n")
                top = df.nlargest(10, "layer2_calibrated_nps")[
                    ["product_line", "data_source", "total_records",
                     "raw_nps", "layer2_calibrated_nps"]
                ]
                f.write("| 品线 | 数据源 | 记录数 | 原始 NPS | 校准后 NPS |\n")
                f.write("|------|--------|--------|---------|-----------|\n")
                for _, row in top.iterrows():
                    f.write(f"| {row['product_line']} | {row['data_source']} | "
                            f"{row['total_records']} | {row['raw_nps']:.1f} | "
                            f"{row['layer2_calibrated_nps']:.1f} |\n")

        print(f"  Markdown 摘要: {md_path}")

        return {
            "csv": str(csv_path),
            "json": str(json_path),
            "md": str(md_path),
        }

    # ── 一键运行 ─────────────────────────────────────────────────

    def run(
        self,
        csv_path: Path,
        output_dir: Path,
        tag: str = "voc_nps",
    ) -> tuple[pd.DataFrame, dict]:
        """执行完整 pipeline"""
        print("=" * 70)
        print("VOC Proxy NPS Pipeline")
        print("=" * 70)

        # Step 1
        df = self.load(csv_path)

        # Step 2
        df = self.compute_raw_nps(df)

        # Step 3
        df = self.apply_source_calibration(df)

        # Step 4
        df = self.apply_category_standardization(df)

        # Step 5
        self.run_audit(df)

        # Step 6
        paths = self.export(df, output_dir, tag)

        print("\n" + "=" * 70)
        print("Pipeline 完成")
        print("=" * 70)

        return df, paths


# ─────────────────────────────────────────────────────────────────
# 辅助：从分类数据构建品类基线
# ─────────────────────────────────────────────────────────────────

def build_category_baseline_from_classified(
    classified_csv: Path,
    category_col: str = "product_line",
    min_samples: int = 10,
) -> dict[str, dict]:
    """从已分类数据计算品类 NPS 基线，供 CategoryStandardizer 使用"""
    df = pd.read_csv(classified_csv)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df["rating"] = df["rating"].astype(int)

    if category_col not in df.columns:
        raise ValueError(f"品类列 '{category_col}' 不在数据中，可用列: {list(df.columns)}")

    category_stats = {}
    for cat in df[category_col].dropna().unique():
        sub = df[df[category_col] == cat]
        if len(sub) < min_samples:
            continue
        promoter = (sub["rating"] >= 4).sum()
        detractor = (sub["rating"] <= 2).sum()
        nps = (promoter - detractor) / len(sub) * 100
        category_stats[cat] = {
            "avg_nps": round(nps, 2),
            "sample_size": len(sub),
        }

    return category_stats


# ─────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base_dir = Path(__file__).parent
    classified_csv = base_dir / "classified" / "momcozy_classified.csv"
    output_dir = base_dir / "pipeline_output"

    # 构建品类基线（按 product_line）
    category_stats = build_category_baseline_from_classified(
        classified_csv, category_col="product_line"
    )
    standardizer = CategoryStandardizer(category_stats)

    # 初始化 pipeline
    pipeline = VOCNPSPipeline(standardizer=standardizer)

    # 运行
    df, paths = pipeline.run(classified_csv, output_dir, tag="momcozy_voc")

    print(f"\n输出文件:")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    print(f"\n总分组数: {len(df)}")
    print(f"审计状态: {pipeline.audit_report['status']}")
    print(f"\n前 5 行结果:")
    print(df.head())
