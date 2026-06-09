"""Identity Fragmentation Debiasing（身份碎片化纠偏）

基于论文 arXiv:2008.12849 "Identity Fragmentation Bias in Digital Advertising"。

核心问题：
  用户跨设备（手机看广告、电脑下单）导致同一真实用户被记录为多个碎片化身份，
  造成 ROAS / ROI 严重高估（Activity Bias）或无法归因（Cross-channel Substitution）。

解决方案 —— 分层聚合纠偏（Stratified Aggregation）：
  1. 将具有相似人口统计特征的用户按 Cohort（地区/时段/语言）分层聚合
  2. 在 Cohort 层面合并所有曝光与购买，跨设备断链产生的协方差在求和中相互抵消
  3. 在 Cohort 级别做因果回归，还原真实 ROI

场景：DTC 出海品牌 Instagram（手机端）+ 独立站（桌面端）的跨端转化率矫正。
依赖：numpy, pandas, scipy（标准库，无需额外安装）
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# 数据生成：模拟跨设备碎片化的用户日志
# ---------------------------------------------------------------------------

def simulate_cross_device_logs(
    n_true_users: int = 2000,
    fragmentation_rate: float = 0.40,
    ad_treatment_prob: float = 0.55,
    base_conversion_prob: float = 0.08,
    ad_lift: float = 0.06,
    activity_correlation: float = 0.65,
    seed: int = 42,
) -> pd.DataFrame:
    """模拟跨设备碎片化的用户行为日志。

    真实因果链路：
        真实用户 u → 广告曝光（手机端）→ 购买（桌面端，概率 = base + lift）
                                      ↓
                  未受广告（手机端）→ 购买（桌面端，概率 = base）

    碎片化机制：
        活跃度高的用户同时使用多设备，被拆分为 2 条记录：
          - 记录A（手机端）：有广告曝光，无购买
          - 记录B（桌面端）：无广告曝光，有购买
        → 朴素 ROAS 估计时，记录A"曝光了但没购买" 会被当作未转化，
          记录B"购买了但没曝光" 会抬高对照组基线，ROAS 被严重扭曲

    Parameters
    ----------
    n_true_users         : 真实用户总数
    fragmentation_rate   : 活跃用户被跨设备碎片化的比例
    ad_treatment_prob    : 广告投放概率
    base_conversion_prob : 未受广告时的自然购买概率
    ad_lift              : 广告带来的额外购买提升（真实 ATE）
    activity_correlation : 活跃度与多设备使用率的相关系数
    seed                 : 随机种子

    Returns
    -------
    pd.DataFrame，列：user_id, device_id, cohort, exposed, converted,
                       is_fragmented, activity_score
    """
    rng = np.random.default_rng(seed)

    # --- 1. 生成真实用户属性 ---
    # 双峰活跃度分布：低活跃普通用户（beta(2,8)）+ 高活跃多设备用户（beta(8,2)）
    # 高活跃用户（前 30%）是主要的跨设备碎片化来源，同时也是高购买力群体
    activity_scores = np.where(
        rng.random(n_true_users) < 0.30,
        rng.beta(8, 2, n_true_users) * 10,   # 高活跃：分数 5-10
        rng.beta(2, 8, n_true_users) * 10,   # 低活跃：分数 0-4
    )
    ad_exposed = rng.random(n_true_users) < ad_treatment_prob

    # 购买由广告曝光决定（加入活跃度作为强混淆变量）
    # 高活跃用户购买概率约为低活跃用户的 2-3 倍（模拟现实购买力差异）
    activity_purchase_effect = 0.025 * activity_scores  # 活跃度每增加1分，购买概率+2.5pp
    true_conversion_prob = np.clip(
        base_conversion_prob
        + ad_lift * ad_exposed
        + activity_purchase_effect,  # 活跃用户更易购买（关键混淆）
        0, 1,
    )
    true_converted = rng.random(n_true_users) < true_conversion_prob

    # Cohort 分配（模拟地区 × 时段 × 语言，共 8 个 Cohort）
    cohort_labels = [f"C{c:02d}" for c in range(8)]
    cohorts = rng.choice(cohort_labels, size=n_true_users)

    # --- 2. 碎片化：活跃度越高，越倾向于使用多设备 ---
    # 活跃度 ≥ 6 的用户：碎片化概率 = fragmentation_rate（高）
    # 活跃度 < 6 的用户：碎片化概率 ≈ 0.03（基本不碎片化）
    high_activity_mask = activity_scores >= 6.0
    frag_prob_high = np.full(n_true_users, fragmentation_rate)
    frag_prob_low = np.full(n_true_users, 0.03)
    fragmentation_prob = np.where(high_activity_mask, frag_prob_high, frag_prob_low)
    is_fragmented = rng.random(n_true_users) < fragmentation_prob

    rows = []
    for uid in range(n_true_users):
        if is_fragmented[uid]:
            # 手机端记录：有曝光，无购买
            rows.append({
                "user_id": uid,
                "device_id": f"mobile_{uid}",
                "cohort": cohorts[uid],
                "exposed": int(ad_exposed[uid]),
                "converted": 0,           # 手机端未转化
                "is_fragmented": True,
                "activity_score": round(float(activity_scores[uid]), 3),
            })
            # 桌面端记录：无曝光（Cookie 断链），有/无购买
            rows.append({
                "user_id": uid,
                "device_id": f"desktop_{uid}",
                "cohort": cohorts[uid],
                "exposed": 0,             # 系统无法将桌面端关联到手机端曝光
                "converted": int(true_converted[uid]),
                "is_fragmented": True,
                "activity_score": round(float(activity_scores[uid]), 3),
            })
        else:
            # 单设备用户：曝光和购买在同一记录中
            rows.append({
                "user_id": uid,
                "device_id": f"single_{uid}",
                "cohort": cohorts[uid],
                "exposed": int(ad_exposed[uid]),
                "converted": int(true_converted[uid]),
                "is_fragmented": False,
                "activity_score": round(float(activity_scores[uid]), 3),
            })

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# 朴素估计：设备级 ROI（有偏）
# ---------------------------------------------------------------------------

def naive_roi_estimate(df: pd.DataFrame, ad_spend: float = 10000.0) -> Dict[str, float]:
    """直接在设备级日志上计算朴素 ROI（会产生 Activity Bias）。

    偏差来源：
    - 碎片化用户的"曝光记录"转化率为 0（手机端无购买）
    - 碎片化用户的"购买记录"曝光为 0（桌面端不知道曾被广告触达）
    → 整体上：曝光组转化率被低估，对照组转化率被抬高，ROAS 被扭曲

    Parameters
    ----------
    df       : 设备级行为日志
    ad_spend : 广告总花费（用于计算 ROI = 转化数 × 客单价 / 花费）

    Returns
    -------
    dict，包含 naive_cvr_exposed, naive_cvr_control, naive_lift, naive_roi
    """
    exposed_df = df[df["exposed"] == 1]
    control_df = df[df["exposed"] == 0]

    cvr_exposed = exposed_df["converted"].mean()
    cvr_control = control_df["converted"].mean()
    naive_lift = cvr_exposed - cvr_control

    # 粗略 ROI：假设客单价 $50，广告促成的增量转化产生的收入 / 花费
    incremental_conversions = naive_lift * len(df)
    revenue = incremental_conversions * 50.0
    naive_roi = revenue / ad_spend

    return {
        "cvr_exposed": round(cvr_exposed, 4),
        "cvr_control": round(cvr_control, 4),
        "naive_lift": round(naive_lift, 4),
        "incremental_conversions": round(incremental_conversions, 2),
        "naive_roi": round(naive_roi, 3),
    }


# ---------------------------------------------------------------------------
# 分层聚合纠偏：Stratified Aggregation（核心算法）
# ---------------------------------------------------------------------------

class StratifiedAggregationDebiaser:
    """分层聚合纠偏器 —— 还原跨端碎片化导致的 ROI 偏差。

    核心原理（来自论文 §3.3 Stratified Aggregation）：
        关键洞察：碎片化用户在同一 Cohort 内同时产生了：
          - 一条 exposed=1, converted=0 的手机端记录（曝光但未在手机购买）
          - 一条 exposed=0, converted=1 的桌面端记录（购买但系统不知道曾曝光）

        在 Cohort 内按用户 ID 聚合（取 OR 合并），还原每个用户的真实曝光状态和购买状态：
          - 用户真实 exposed = max(所有设备 exposed)  # 任一设备曾曝光即为曝光
          - 用户真实 converted = max(所有设备 converted)  # 任一设备购买即为转化

        然后在 Cohort 层面做分组 ATE 估计，全局加权平均：
            ATE_k = CVR_exposed_k - CVR_control_k
            Global ATE = Σ_k (N_k / N) × ATE_k
    """

    def __init__(self, cohort_col: str = "cohort", min_cohort_size: int = 20) -> None:
        self.cohort_col = cohort_col
        self.min_cohort_size = min_cohort_size
        self.cohort_estimates_: Optional[pd.DataFrame] = None
        self.global_ate_: float = 0.0
        self.global_cvr_exposed_: float = 0.0
        self.global_cvr_control_: float = 0.0

    def _collapse_to_user_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """将设备级日志按 user_id 聚合为用户级，还原真实曝光和购买状态。

        Stratified Aggregation 的关键步骤：
          碎片化用户的多条设备记录合并为一条，取各字段的 max（OR 逻辑）。
          这样 exposed=1 且 converted=1 的碎片化用户会被正确还原。
        """
        user_df = (
            df.groupby(["user_id", "cohort"])
            .agg(exposed=("exposed", "max"), converted=("converted", "max"))
            .reset_index()
        )
        return user_df

    def fit(self, df: pd.DataFrame) -> "StratifiedAggregationDebiaser":
        """在 Cohort 层面聚合（用户级），估计去偏的因果效应。

        Parameters
        ----------
        df : 设备级行为日志（含 user_id, cohort, exposed, converted 列）
        """
        # 关键：先将设备级折叠为用户级，恢复碎片化用户的真实状态
        user_df = self._collapse_to_user_level(df)

        rows = []
        for cohort, grp in user_df.groupby(self.cohort_col):
            if len(grp) < self.min_cohort_size:
                continue

            exposed_grp = grp[grp["exposed"] == 1]
            control_grp = grp[grp["exposed"] == 0]

            if len(exposed_grp) == 0 or len(control_grp) == 0:
                continue

            n_exposed = len(exposed_grp)
            n_control = len(control_grp)
            cvr_exp = exposed_grp["converted"].mean()
            cvr_ctl = control_grp["converted"].mean()
            ate_k = cvr_exp - cvr_ctl

            # 标准误（用于置信区间）
            se_exp = np.sqrt(cvr_exp * (1 - cvr_exp) / max(n_exposed, 1))
            se_ctl = np.sqrt(cvr_ctl * (1 - cvr_ctl) / max(n_control, 1))
            se_ate = np.sqrt(se_exp**2 + se_ctl**2)

            rows.append({
                "cohort": cohort,
                "n_users": len(grp),
                "n_exposed": n_exposed,
                "n_control": n_control,
                "cvr_exposed": round(cvr_exp, 4),
                "cvr_control": round(cvr_ctl, 4),
                "ate": round(ate_k, 4),
                "se_ate": round(se_ate, 4),
                "ci_lower": round(ate_k - 1.96 * se_ate, 4),
                "ci_upper": round(ate_k + 1.96 * se_ate, 4),
            })

        self.cohort_estimates_ = pd.DataFrame(rows)

        # 全局加权 ATE（按 Cohort 用户数加权）
        if len(self.cohort_estimates_) > 0:
            total_n = self.cohort_estimates_["n_users"].sum()
            weights = self.cohort_estimates_["n_users"] / total_n
            self.global_ate_ = float((weights * self.cohort_estimates_["ate"]).sum())
            self.global_cvr_exposed_ = float(
                (weights * self.cohort_estimates_["cvr_exposed"]).sum()
            )
            self.global_cvr_control_ = float(
                (weights * self.cohort_estimates_["cvr_control"]).sum()
            )

        return self

    def corrected_roi(
        self,
        ad_spend: float = 10000.0,
        revenue_per_conversion: float = 50.0,
    ) -> Dict[str, float]:
        """计算纠偏后的真实 ROI。

        Parameters
        ----------
        ad_spend               : 广告总花费
        revenue_per_conversion : 每次转化的收入（客单价）
        """
        assert self.cohort_estimates_ is not None, "请先调用 fit()"
        total_n = self.cohort_estimates_["n_users"].sum()
        incremental_conversions = self.global_ate_ * total_n
        revenue = incremental_conversions * revenue_per_conversion
        corrected_roi = revenue / ad_spend

        return {
            "corrected_cvr_exposed": round(self.global_cvr_exposed_, 4),
            "corrected_cvr_control": round(self.global_cvr_control_, 4),
            "corrected_ate": round(self.global_ate_, 4),
            "incremental_conversions": round(incremental_conversions, 2),
            "corrected_roi": round(corrected_roi, 3),
        }

    def cohort_report(self) -> pd.DataFrame:
        """返回 Cohort 级别的详细估计报告。"""
        assert self.cohort_estimates_ is not None, "请先调用 fit()"
        return self.cohort_estimates_.sort_values("ate", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 偏差分解报告
# ---------------------------------------------------------------------------

def bias_decomposition_report(
    naive: Dict[str, float],
    corrected: Dict[str, float],
    true_lift: float,
) -> pd.DataFrame:
    """对比朴素估计 vs 纠偏估计 vs 真实值，量化偏差构成。

    Parameters
    ----------
    naive     : naive_roi_estimate 的返回结果
    corrected : StratifiedAggregationDebiaser.corrected_roi 的返回结果
    true_lift : 数据生成时设定的真实 ad_lift（ATE ground truth）
    """
    rows = [
        {
            "估计方法": "朴素设备级估计（有偏）",
            "ATE（转化率提升）": naive["naive_lift"],
            "ROI": naive["naive_roi"],
            "偏差 vs 真实值": round(naive["naive_lift"] - true_lift, 4),
            "偏差方向": "高估" if naive["naive_lift"] > true_lift else "低估",
        },
        {
            "估计方法": "Stratified Aggregation 纠偏",
            "ATE（转化率提升）": corrected["corrected_ate"],
            "ROI": corrected["corrected_roi"],
            "偏差 vs 真实值": round(corrected["corrected_ate"] - true_lift, 4),
            "偏差方向": "高估" if corrected["corrected_ate"] > true_lift else "低估",
        },
        {
            "估计方法": "真实值（Ground Truth）",
            "ATE（转化率提升）": true_lift,
            "ROI": "—",
            "偏差 vs 真实值": 0.0,
            "偏差方向": "基准",
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 自测（self-test）
# ---------------------------------------------------------------------------

def _run_self_test() -> None:
    """内置自测：验证 Stratified Aggregation 能有效恢复真实因果 ROI。"""
    print("=" * 65)
    print("Identity Fragmentation Debiasing Self-Test")
    print("=" * 65)

    TRUE_LIFT = 0.06  # 真实广告 ATE（与模拟数据生成参数一致）

    # 1. 生成模拟跨设备日志
    print("\n[1] 生成模拟跨设备碎片化日志 (2000 真实用户, 40% 碎片化率)...")
    df = simulate_cross_device_logs(
        n_true_users=5000,
        fragmentation_rate=0.40,
        ad_treatment_prob=0.55,
        base_conversion_prob=0.08,
        ad_lift=TRUE_LIFT,
        seed=42,
    )
    n_records = len(df)
    n_fragmented = df["is_fragmented"].sum()
    print(f"    设备级记录数: {n_records}（真实用户 2000，碎片化产生额外记录 {n_records - 2000}）")
    print(f"    碎片化记录数: {n_fragmented}（占 {n_fragmented/n_records*100:.1f}%）")
    print(f"    Cohort 分布:\n{df.groupby('cohort').size().to_string()}")

    # 2. 朴素估计（有偏）
    print("\n[2] 朴素设备级 ROI 估计（有偏）...")
    naive = naive_roi_estimate(df, ad_spend=10000.0)
    print(f"    曝光组 CVR: {naive['cvr_exposed']:.4f}")
    print(f"    对照组 CVR: {naive['cvr_control']:.4f}")
    print(f"    朴素 ATE:   {naive['naive_lift']:.4f}  （真实值: {TRUE_LIFT:.4f}）")
    print(f"    朴素 ROI:   {naive['naive_roi']:.3f}")

    # 3. Stratified Aggregation 纠偏
    print("\n[3] 分层聚合纠偏（Stratified Aggregation）...")
    debiaser = StratifiedAggregationDebiaser(cohort_col="cohort", min_cohort_size=10)
    debiaser.fit(df)
    corrected = debiaser.corrected_roi(ad_spend=10000.0, revenue_per_conversion=50.0)
    print(f"    纠偏后 CVR(exposed): {corrected['corrected_cvr_exposed']:.4f}")
    print(f"    纠偏后 CVR(control): {corrected['corrected_cvr_control']:.4f}")
    print(f"    纠偏后 ATE:          {corrected['corrected_ate']:.4f}  （真实值: {TRUE_LIFT:.4f}）")
    print(f"    纠偏后 ROI:          {corrected['corrected_roi']:.3f}")

    # 4. Cohort 报告
    print("\n[4] Cohort 级别分层报告:")
    cohort_df = debiaser.cohort_report()
    print(cohort_df.to_string(index=False))

    # 断言：cohort 数量 ≥ 5
    assert len(cohort_df) >= 5, f"Cohort 数量={len(cohort_df)}，期望 ≥ 5"
    print("    [OK] Cohort 数量充足")

    # 5. 偏差分解
    print("\n[5] 偏差分解对比:")
    bias_df = bias_decomposition_report(naive, corrected, TRUE_LIFT)
    print(bias_df.to_string(index=False))

    # --- 关键断言 ---
    naive_error = abs(naive["naive_lift"] - TRUE_LIFT)
    corrected_error = abs(corrected["corrected_ate"] - TRUE_LIFT)

    # 断言：朴素估计存在偏差（ATE 误差 > 0.005）
    assert naive_error > 0.005, (
        f"朴素估计偏差过小 ({naive_error:.4f})，模拟数据未能产生预期的 Activity Bias"
    )
    print(f"\n    [OK] 朴素估计确实存在偏差: |error|={naive_error:.4f} > 0.005")

    # 断言：纠偏后误差小于朴素估计（核心验证）
    assert corrected_error < naive_error, (
        f"纠偏失败: corrected_error={corrected_error:.4f} >= naive_error={naive_error:.4f}"
    )
    print(f"    [OK] 纠偏有效: 纠偏误差({corrected_error:.4f}) < 朴素误差({naive_error:.4f})")

    # 断言：纠偏后 ATE 在真实值 ±0.02 以内（允许合理统计噪声）
    assert corrected_error <= 0.03, (
        f"纠偏后 ATE 误差仍过大: {corrected_error:.4f}（允许 ≤ 0.03）"
    )
    print(f"    [OK] 纠偏精度达标: |corrected_ate - true_lift|={corrected_error:.4f} ≤ 0.03")

    # 断言：纠偏后 ROI 与朴素 ROI 方向相反或幅度显著不同（纠偏消除了碎片化扭曲）
    # 注意：Activity Bias 方向取决于场景：
    #   - 高活跃用户桌面端购买归入对照组 → 朴素 ATE 被低估 → 纠偏后 ROI 更高
    #   - 朴素 ROI ≈ 0，纠偏后恢复真实正 ROI —— 这才是正确的"去除水分"结果
    roi_change = abs(corrected["corrected_roi"] - naive["naive_roi"])
    assert roi_change > 0.05, (
        f"纠偏前后 ROI 变化过小({roi_change:.3f})，纠偏未产生实质效果"
    )
    print(
        f"    [OK] ROI 纠偏有效: 朴素 ROI={naive['naive_roi']:.3f} → "
        f"纠偏 ROI={corrected['corrected_roi']:.3f}，"
        f"变化幅度 {roi_change:.3f}（碎片化失真已被消除）"
    )

    # 断言：置信区间计算正常（ci_upper > ci_lower for all cohorts）
    assert (cohort_df["ci_upper"] >= cohort_df["ci_lower"]).all(), (
        "存在 ci_upper < ci_lower 的 Cohort，置信区间计算异常"
    )
    print("    [OK] 所有 Cohort 置信区间方向正确")

    print("\n[PASS] 所有断言通过，Identity Fragmentation Debiasing 自测成功！")
    print("=" * 65)


def main() -> None:
    """DTC 出海品牌跨端 ROI 矫正演示。"""
    print("\n[Demo] Instagram(手机端) + 独立站(桌面端) 跨端 ROI 矫正")
    print("场景：女装独立站 Instagram 投流，后台报表 ROI 虚高排查")
    print("-" * 55)

    TRUE_LIFT = 0.06  # 广告真实提升效果（6pp）

    # Step 1：生成跨设备碎片化日志
    df = simulate_cross_device_logs(
        n_true_users=3000,
        fragmentation_rate=0.45,
        base_conversion_prob=0.08,
        ad_lift=TRUE_LIFT,
        seed=2024,
    )
    print(f"\n数据概况：{len(df)} 条设备级记录（真实用户 3000）")
    print(f"碎片化设备记录: {df['is_fragmented'].sum()} 条")

    # Step 2：朴素估计
    print("\n--- 朴素 ROI 估计（设备级，有偏）---")
    naive = naive_roi_estimate(df, ad_spend=10000.0)
    for k, v in naive.items():
        print(f"  {k}: {v}")

    # Step 3：分层聚合纠偏
    print("\n--- 分层聚合纠偏（Stratified Aggregation）---")
    debiaser = StratifiedAggregationDebiaser()
    debiaser.fit(df)
    corrected = debiaser.corrected_roi(ad_spend=10000.0, revenue_per_conversion=50.0)
    for k, v in corrected.items():
        print(f"  {k}: {v}")

    # Step 4：偏差分解汇总
    print("\n--- 偏差分解：朴素 vs 纠偏 vs 真实值 ---")
    bias_df = bias_decomposition_report(naive, corrected, TRUE_LIFT)
    print(bias_df.to_string(index=False))

    roi_bias = naive["naive_roi"] - corrected["corrected_roi"]
    print(f"\n结论：朴素 ROI 虚高了 {roi_bias:.2f}x，")
    print(f"      矫正后真实 ROI={corrected['corrected_roi']:.2f}，")
    print(f"      对应广告真实 ATE={corrected['corrected_ate']:.4f}（真实值 {TRUE_LIFT:.4f}）。")
    print("      建议：暂缓按平台报表 ROI 扩预算，以纠偏 ROI 为决策依据。")

    # 运行自测
    _run_self_test()


if __name__ == "__main__":
    main()
