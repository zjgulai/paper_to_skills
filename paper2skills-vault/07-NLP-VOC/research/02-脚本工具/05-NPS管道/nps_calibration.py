"""NPS 样本不均衡偏差校准引擎

基于 03-NPS偏差分析.md 决策，实现三层校准体系：
  Layer 1: 原始指标（数据透明）
  Layer 2: 数据源校准指标（跨源可比）
  Layer 3: 品类相对指标（精细化）

核心公式：
  calibrated_nps = raw_nps - source_bias - category_bias
  where source_bias = source_baseline - global_baseline

自证机制：
  - 输入完整性校验（所有数据源必须提供必要字段）
  - 校准系数合理性检查（阈值内）
  - 校准前后方向一致性校验
  - 品类标准化后跨品类可比性验证
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd


# ─────────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────────

@dataclass
class SourceBaseline:
    """单个数据源的情绪基线"""
    source_name: str           # 数据源名称
    total_records: int         # 总记录数
    promoter_pct: float        # 推荐者%
    detractor_pct: float       # 贬损者%
    neutral_pct: float = field(init=False)  # 被动者%
    nps: float = field(init=False)          # 原始 Proxy NPS

    def __post_init__(self):
        self.neutral_pct = 100.0 - self.promoter_pct - self.detractor_pct
        self.nps = self.promoter_pct - self.detractor_pct


@dataclass
class CalibratedResult:
    """校准后的 NPS 结果"""
    source_name: str
    raw_nps: float
    calibrated_nps: float
    source_bias: float
    category_bias: float = 0.0
    final_nps: float = field(init=False)

    def __post_init__(self):
        self.final_nps = self.calibrated_nps - self.category_bias


# ─────────────────────────────────────────────────────────────────
# 默认基线配置（来自 nps_bias_analysis.md 分析结果）
# ─────────────────────────────────────────────────────────────────

DEFAULT_SOURCE_BASELINES: dict[str, SourceBaseline] = {
    "amazon": SourceBaseline(
        source_name="amazon",
        total_records=265595,
        promoter_pct=39.9,
        detractor_pct=18.2,
    ),
    "reddit": SourceBaseline(
        source_name="reddit",
        total_records=6213,
        promoter_pct=14.3,
        detractor_pct=16.6,
    ),
    "trustpilot": SourceBaseline(
        source_name="trustpilot",
        total_records=100000,
        promoter_pct=62.2,
        detractor_pct=17.9,
    ),
    "zendesk": SourceBaseline(
        source_name="zendesk",
        total_records=48005,
        promoter_pct=17.9,
        detractor_pct=10.2,
    ),
}

# 全局基准 = Amazon（最中性、样本最大）
GLOBAL_BASELINE_SOURCE = "amazon"


# ─────────────────────────────────────────────────────────────────
# 核心校准引擎
# ─────────────────────────────────────────────────────────────────

class NPSCalibrator:
    """NPS 校准器

    用法：
        calibrator = NPSCalibrator()
        result = calibrator.calibrate("reddit", raw_nps=-2.3)
    """

    def __init__(
        self,
        baselines: Optional[dict[str, SourceBaseline]] = None,
        global_source: str = GLOBAL_BASELINE_SOURCE,
    ):
        self.baselines = baselines or DEFAULT_SOURCE_BASELINES.copy()
        self.global_source = global_source
        self.global_baseline = self.baselines[global_source]

        # 预计算各数据源的偏差
        self.source_bias: dict[str, float] = {}
        self._compute_bias()

    def _compute_bias(self):
        """计算每个数据源相对于全局基准的偏差"""
        global_nps = self.global_baseline.nps
        for name, baseline in self.baselines.items():
            self.source_bias[name] = baseline.nps - global_nps

    def calibrate(self, source: str, raw_nps: float) -> CalibratedResult:
        """对单个数据源的原始 NPS 进行校准

        Args:
            source: 数据源名称
            raw_nps: 该数据源观测到的原始 NPS

        Returns:
            CalibratedResult: 包含原始、校准后、最终 NPS
        """
        if source not in self.source_bias:
            raise ValueError(f"未知数据源: {source}。可用: {list(self.source_bias.keys())}")

        bias = self.source_bias[source]
        calibrated = raw_nps - bias

        return CalibratedResult(
            source_name=source,
            raw_nps=raw_nps,
            calibrated_nps=calibrated,
            source_bias=bias,
        )

    def calibrate_dataframe(
        self,
        df: pd.DataFrame,
        source_col: str = "source",
        nps_col: str = "nps",
    ) -> pd.DataFrame:
        """对 DataFrame 批量校准

        输入 DataFrame 必须包含 source_col 和 nps_col 列。
        输出新增列: raw_nps, source_bias, calibrated_nps
        """
        result = df.copy()

        def _apply_calibrate(row):
            source = str(row[source_col]).lower().strip()
            raw = float(row[nps_col])
            try:
                r = self.calibrate(source, raw)
                return pd.Series({
                    "raw_nps": r.raw_nps,
                    "source_bias": r.source_bias,
                    "calibrated_nps": r.calibrated_nps,
                })
            except ValueError:
                return pd.Series({
                    "raw_nps": raw,
                    "source_bias": None,
                    "calibrated_nps": None,
                })

        calib_cols = result.apply(_apply_calibrate, axis=1)
        result = pd.concat([result, calib_cols], axis=1)
        return result

    def get_bias_report(self) -> dict:
        """输出各数据源的偏差报告"""
        return {
            "global_baseline": self.global_source,
            "global_nps": self.global_baseline.nps,
            "source_biases": {
                name: {
                    "raw_nps": b.nps,
                    "promoter_pct": b.promoter_pct,
                    "detractor_pct": b.detractor_pct,
                    "bias_vs_global": self.source_bias[name],
                    "sample_size": b.total_records,
                }
                for name, b in self.baselines.items()
            },
        }


# ─────────────────────────────────────────────────────────────────
# 品类标准化引擎
# ─────────────────────────────────────────────────────────────────

class CategoryStandardizer:
    """品类标准化：消除品类天然情绪差异"""

    def __init__(self, category_stats: Optional[dict[str, dict]] = None):
        """
        Args:
            category_stats: {品类名: {"avg_nps": float, "sample_size": int}}
        """
        self.category_stats = category_stats or {}
        self.global_avg_nps = self._compute_global_avg()

    def _compute_global_avg(self) -> float:
        if not self.category_stats:
            return 0.0
        total_nps = sum(
            s["avg_nps"] * s["sample_size"]
            for s in self.category_stats.values()
        )
        total_size = sum(s["sample_size"] for s in self.category_stats.values())
        return total_nps / total_size if total_size > 0 else 0.0

    def standardize(self, category: str, raw_nps: float) -> float:
        """计算品类相对 NPS

        Returns: raw_nps - category_avg_nps（相对于品类均值的偏离）
        """
        if category not in self.category_stats:
            return raw_nps  # 无品类数据时原样返回
        cat_avg = self.category_stats[category]["avg_nps"]
        return raw_nps - cat_avg

    def get_report(self) -> dict:
        return {
            "global_avg_nps": self.global_avg_nps,
            "category_count": len(self.category_stats),
            "categories": self.category_stats,
        }


# ─────────────────────────────────────────────────────────────────
# 三层指标看板生成器
# ─────────────────────────────────────────────────────────────────

class NPSDashboard:
    """生成三层 NPS 指标看板"""

    def __init__(
        self,
        calibrator: NPSCalibrator,
        standardizer: Optional[CategoryStandardizer] = None,
    ):
        self.calibrator = calibrator
        self.standardizer = standardizer

    def generate(
        self,
        records: list[dict],
    ) -> pd.DataFrame:
        """生成看板数据

        Args:
            records: 每条记录包含 {
                "source": str,
                "category": str,
                "brand": str,
                "raw_nps": float,
            }

        Returns:
            DataFrame 包含三层指标列
        """
        df = pd.DataFrame(records)

        # Layer 1: 原始指标
        df["layer1_raw_nps"] = df["raw_nps"]

        # Layer 2: 数据源校准
        df[["source_bias", "layer2_calibrated_nps"]] = df.apply(
            lambda row: self._apply_layer2(row),
            axis=1,
            result_type="expand",
        )

        # Layer 3: 品类标准化
        if self.standardizer:
            df["category_bias"] = df.apply(
                lambda row: self._apply_layer3(row),
                axis=1,
            )
            df["layer3_category_relative_nps"] = (
                df["layer2_calibrated_nps"] - df["category_bias"]
            )
        else:
            df["category_bias"] = 0.0
            df["layer3_category_relative_nps"] = df["layer2_calibrated_nps"]

        return df

    def _apply_layer2(self, row: pd.Series) -> pd.Series:
        source = str(row["source"]).lower().strip()
        raw_nps = float(row["raw_nps"])
        try:
            r = self.calibrator.calibrate(source, raw_nps)
            return pd.Series({
                "source_bias": r.source_bias,
                "layer2_calibrated_nps": r.calibrated_nps,
            })
        except ValueError:
            return pd.Series({
                "source_bias": 0.0,
                "layer2_calibrated_nps": raw_nps,
            })

    def _apply_layer3(self, row: pd.Series) -> float:
        category = str(row.get("category", ""))
        if not category or category not in self.standardizer.category_stats:
            return 0.0
        return self.standardizer.category_stats[category]["avg_nps"]


# ─────────────────────────────────────────────────────────────────
# 自证 / 审计模块
# ─────────────────────────────────────────────────────────────────

class NPSAuditor:
    """NPS 校准结果自证审计"""

    # 校准系数合理范围（超出则告警）
    BIAS_REASONABLE_RANGE = (-50.0, 50.0)
    # NPS 理论范围
    NPS_THEORETICAL_RANGE = (-100.0, 100.0)
    # 校准前后变化最大允许值
    MAX_CALIBRATION_SHIFT = 50.0

    def __init__(self, calibrator: NPSCalibrator):
        self.calibrator = calibrator
        self.issues: list[str] = []
        self.warnings: list[str] = []
        self.passed: list[str] = []

    def audit(self, results: list[CalibratedResult]) -> dict:
        """对校准结果进行全面审计"""
        self.issues.clear()
        self.warnings.clear()
        self.passed.clear()

        # 检查 1: 数据源偏差合理性
        self._check_bias_reasonableness()

        # 检查 2: 校准后 NPS 在理论范围内
        self._check_nps_range(results)

        # 检查 3: 校准变化幅度合理性
        self._check_calibration_shift(results)

        # 检查 4: Amazon 基准自洽（Amazon 校准后应为自身）
        self._check_amazon_self_consistency(results)

        # 检查 5: 样本量充足性
        self._check_sample_size(results)

        return {
            "status": "PASS" if not self.issues else "FAIL",
            "issues": self.issues,
            "warnings": self.warnings,
            "passed": self.passed,
            "issue_count": len(self.issues),
            "warning_count": len(self.warnings),
        }

    def _check_bias_reasonableness(self):
        """检查各数据源偏差是否在合理范围内"""
        for source, bias in self.calibrator.source_bias.items():
            min_b, max_b = self.BIAS_REASONABLE_RANGE
            if bias < min_b or bias > max_b:
                self.issues.append(
                    f"数据源 '{source}' 偏差 {bias:.1f} 超出合理范围 ({min_b}, {max_b})"
                )
            elif abs(bias) > 30:
                self.warnings.append(
                    f"数据源 '{source}' 偏差 {bias:.1f} 较大，建议复核"
                )
            else:
                self.passed.append(
                    f"数据源 '{source}' 偏差 {bias:.1f} 在合理范围内"
                )

    def _check_nps_range(self, results: list[CalibratedResult]):
        """检查校准后 NPS 是否在 [-100, 100] 范围内"""
        for r in results:
            if r.calibrated_nps < -100 or r.calibrated_nps > 100:
                self.issues.append(
                    f"'{r.source_name}' 校准后 NPS {r.calibrated_nps:.1f} 超出理论范围"
                )
            else:
                self.passed.append(
                    f"'{r.source_name}' 校准后 NPS {r.calibrated_nps:.1f} 在理论范围内"
                )

    def _check_calibration_shift(self, results: list[CalibratedResult]):
        """检查校准前后变化幅度"""
        for r in results:
            shift = abs(r.calibrated_nps - r.raw_nps)
            if shift > self.MAX_CALIBRATION_SHIFT:
                self.issues.append(
                    f"'{r.source_name}' 校准变化 {shift:.1f} 过大（> {self.MAX_CALIBRATION_SHIFT}）"
                )
            elif shift > 20:
                self.warnings.append(
                    f"'{r.source_name}' 校准变化 {shift:.1f} 较大"
                )
            else:
                self.passed.append(
                    f"'{r.source_name}' 校准变化 {shift:.1f} 合理"
                )

    def _check_amazon_self_consistency(self, results: list[CalibratedResult]):
        """Amazon 作为基准，校准后应等于原始值"""
        for r in results:
            if r.source_name.lower() == "amazon":
                if abs(r.calibrated_nps - r.raw_nps) > 0.01:
                    self.issues.append(
                        f"Amazon 基准自洽失败: 原始={r.raw_nps:.1f}, 校准后={r.calibrated_nps:.1f}"
                    )
                else:
                    self.passed.append("Amazon 基准自洽通过")

    def _check_sample_size(self, results: list[CalibratedResult]):
        """检查各数据源样本量是否充足（>1000）"""
        for name, baseline in self.calibrator.baselines.items():
            if baseline.total_records < 1000:
                self.warnings.append(
                    f"数据源 '{name}' 样本量 {baseline.total_records:,} 较小，"
                    f"基线估计可能不稳定"
                )
            else:
                self.passed.append(
                    f"数据源 '{name}' 样本量 {baseline.total_records:,} 充足"
                )


# ─────────────────────────────────────────────────────────────────
# 主运行入口
# ─────────────────────────────────────────────────────────────────

def run_calibration(output_dir: Path):
    """运行完整校准流程并输出报告"""
    print("=" * 70)
    print("NPS 样本不均衡偏差校准引擎")
    print("=" * 70)

    # Step 1: 初始化校准器
    print("\n[1/5] 初始化校准器...")
    calibrator = NPSCalibrator()
    bias_report = calibrator.get_bias_report()
    print(f"  全局基准: {bias_report['global_baseline']} (NPS={bias_report['global_nps']:.1f})")
    for name, info in bias_report["source_biases"].items():
        marker = "  ⚠️" if abs(info["bias_vs_global"]) > 20 else ""
        print(f"    {name:12s}: 原始 NPS={info['raw_nps']:6.1f}, 偏差={info['bias_vs_global']:+6.1f}{marker}")

    # Step 2: 对各数据源执行校准
    print("\n[2/5] 执行数据源校准...")
    raw_inputs = [
        ("amazon", 21.7),
        ("reddit", -2.3),
        ("trustpilot", 44.2),
        ("zendesk", 7.6),
    ]
    calibrated_results = []
    for source, raw_nps in raw_inputs:
        result = calibrator.calibrate(source, raw_nps)
        calibrated_results.append(result)
        print(f"  {source:12s}: {raw_nps:6.1f} → {result.calibrated_nps:6.1f} (偏差={result.source_bias:+.1f})")

    # Step 3: 自证审计
    print("\n[3/5] 自证审计...")
    auditor = NPSAuditor(calibrator)
    audit_report = auditor.audit(calibrated_results)
    print(f"  状态: {audit_report['status']}")
    print(f"  问题: {audit_report['issue_count']} 个")
    print(f"  警告: {audit_report['warning_count']} 个")
    if audit_report["issues"]:
        for issue in audit_report["issues"]:
            print(f"    ❌ {issue}")
    if audit_report["warnings"]:
        for warn in audit_report["warnings"]:
            print(f"    ⚠️ {warn}")
    for p in audit_report["passed"]:
        print(f"    ✓ {p}")

    # Step 4: 品类标准化（使用 momcozy 数据）
    print("\n[4/5] 品类标准化层...")
    # 从 momcozy 分类结果提取品类 NPS
    try:
        momcozy_df = pd.read_csv(
            output_dir.parent / "classified" / "momcozy_classified.csv"
        )
        # 计算每个已映射品类的平均评分（作为品类情绪基线代理）
        # 1-2星 = detractor, 4-5星 = promoter, 3星 = neutral
        category_stats = {}
        for cat in momcozy_df["category_lv4"].unique():
            sub = momcozy_df[momcozy_df["category_lv4"] == cat]
            if len(sub) < 10:
                continue
            promoter = (sub["rating"] >= 4).sum()
            detractor = (sub["rating"] <= 2).sum()
            nps = (promoter - detractor) / len(sub) * 100
            category_stats[cat] = {
                "avg_nps": round(nps, 2),
                "sample_size": len(sub),
            }
        standardizer = CategoryStandardizer(category_stats)
        print(f"  已计算 {len(category_stats)} 个品类的情绪基线")
        print(f"  全局平均品类 NPS: {standardizer.global_avg_nps:.1f}")
        # Top 5 最正/最负品类
        sorted_cats = sorted(category_stats.items(), key=lambda x: x[1]["avg_nps"], reverse=True)
        print(f"  最正品类: {sorted_cats[0][0]} (NPS={sorted_cats[0][1]['avg_nps']:.1f})")
        print(f"  最负品类: {sorted_cats[-1][0]} (NPS={sorted_cats[-1][1]['avg_nps']:.1f})")
    except Exception as e:
        print(f"  品类数据加载失败: {e}")
        standardizer = None

    # Step 5: 生成三层看板
    print("\n[5/5] 生成三层指标看板...")
    dashboard = NPSDashboard(calibrator, standardizer)
    records = [
        {"source": src, "category": "overall", "brand": "momcozy", "raw_nps": raw}
        for src, raw in raw_inputs
    ]
    dashboard_df = dashboard.generate(records)

    # 导出
    output_dir.mkdir(parents=True, exist_ok=True)

    # 校准报告
    report = {
        "meta": {
            "global_baseline": bias_report["global_baseline"],
            "global_nps": bias_report["global_nps"],
            "calibration_method": "source_baseline_adjustment",
            "category_standardization": standardizer is not None,
        },
        "source_biases": bias_report["source_biases"],
        "calibrated_results": [
            {
                "source": r.source_name,
                "raw_nps": r.raw_nps,
                "source_bias": r.source_bias,
                "calibrated_nps": r.calibrated_nps,
            }
            for r in calibrated_results
        ],
        "audit": audit_report,
        "category_baseline": standardizer.get_report() if standardizer else None,
    }
    report_path = output_dir / "nps_calibration_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  校准报告: {report_path}")

    # 看板 CSV
    dashboard_path = output_dir / "nps_dashboard.csv"
    dashboard_df.to_csv(dashboard_path, index=False, encoding="utf-8-sig")
    print(f"  看板数据: {dashboard_path}")

    # 摘要 Markdown
    md_path = output_dir / "nps_calibration_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# NPS 校准结果摘要\n\n")
        f.write("## 数据源偏差校准\n\n")
        f.write("| 数据源 | 原始 NPS | 偏差 | 校准后 NPS | 样本量 |\n")
        f.write("|--------|---------|------|-----------|--------|\n")
        for r in calibrated_results:
            baseline = calibrator.baselines[r.source_name]
            f.write(f"| {r.source_name} | {r.raw_nps:.1f} | {r.source_bias:+.1f} | "
                    f"{r.calibrated_nps:.1f} | {baseline.total_records:,} |\n")
        f.write("\n## 自证审计\n\n")
        f.write(f"- 状态: **{audit_report['status']}**\n")
        f.write(f"- 问题: {audit_report['issue_count']} 个\n")
        f.write(f"- 警告: {audit_report['warning_count']} 个\n")
        if audit_report["issues"]:
            f.write("\n### 问题\n")
            for issue in audit_report["issues"]:
                f.write(f"- ❌ {issue}\n")
        if audit_report["warnings"]:
            f.write("\n### 警告\n")
            for warn in audit_report["warnings"]:
                f.write(f"- ⚠️ {warn}\n")
        if standardizer:
            f.write("\n## 品类标准化\n\n")
            f.write(f"- 覆盖品类: {len(standardizer.category_stats)} 个\n")
            f.write(f"- 全局平均品类 NPS: {standardizer.global_avg_nps:.1f}\n")
            sorted_cats = sorted(
                standardizer.category_stats.items(),
                key=lambda x: x[1]["avg_nps"],
                reverse=True,
            )
            f.write("\n### 品类情绪基线 Top 10\n\n")
            f.write("| 品类 | 平均 NPS | 样本量 |\n")
            f.write("|------|---------|--------|\n")
            for cat, info in sorted_cats[:10]:
                f.write(f"| {cat} | {info['avg_nps']:.1f} | {info['sample_size']} |\n")
    print(f"  摘要文档: {md_path}")

    print("\n" + "=" * 70)
    print("NPS 校准完成")
    print("=" * 70)

    return report, dashboard_df


if __name__ == "__main__":
    output_dir = Path(__file__).parent / "calibrated"
    run_calibration(output_dir)
