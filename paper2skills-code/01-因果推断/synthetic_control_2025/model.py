"""ClusterSC: Advancing Synthetic Control with Donor Clustering.

基于论文 arXiv:2503.21629 的最小骨架实现。
核心思路：
  1. K-Means 对供体池 (donor pool) 历史特征聚类，压缩维度。
  2. 为目标单元（treatment unit）选择距离最近的簇作为缩减供体池。
  3. 在缩减供体池内用约束最小二乘（NNLS + 权重归一化）求合成权重。
  4. 计算处理效应（ATT）= 干预后实际值 - 合成反事实值。

依赖：numpy, pandas, scikit-learn（标准库，无需额外安装）
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")


# ---------------------------------------------------------------------------
# 数据生成（模拟母婴品牌地理级广告投放场景）
# ---------------------------------------------------------------------------

def simulate_geo_data(
    n_donors: int = 200,
    n_treated: int = 1,
    n_pre: int = 24,
    n_post: int = 6,
    treatment_effect: float = 15.0,
    seed: int = 42,
) -> tuple:
    rng = np.random.default_rng(seed)

    pre_cols = [f"T0_{t}" for t in range(n_pre)]
    post_cols = [f"T1_{t}" for t in range(n_post)]
    all_cols = pre_cols + post_cols

    n_total = n_donors + n_treated
    base_trend = np.linspace(100, 130, n_pre + n_post)
    noise = rng.normal(0, 3, (n_total, n_pre + n_post))
    individual_level = rng.normal(0, 5, n_total)[:, None]

    data = base_trend[None, :] + individual_level + noise
    data[-n_treated:, n_pre:] += treatment_effect

    unit_ids = [f"donor_{i:03d}" for i in range(n_donors)] + [
        f"treated_{i:03d}" for i in range(n_treated)
    ]
    df = pd.DataFrame(data, index=unit_ids, columns=all_cols)

    donors = [f"donor_{i:03d}" for i in range(n_donors)]
    treated = [f"treated_{i:03d}" for i in range(n_treated)]
    return df, donors, treated, pre_cols, post_cols


# ---------------------------------------------------------------------------
# 核心 ClusterSC 类
# ---------------------------------------------------------------------------

class ClusterSC:
    """ClusterSC：基于供体聚类的合成控制法 (arXiv:2503.21629)。

    参数
    ----
    n_clusters  : K-Means 聚类数量，默认 10。
    random_state: 随机种子。
    """

    def __init__(self, n_clusters: int = 10, random_state: int = 42) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state

        self._kmeans: KMeans | None = None
        self._weights: dict[str, np.ndarray] = {}
        self._donor_ids_per_unit: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # 步骤 1：供体聚类
    # ------------------------------------------------------------------

    def _cluster_donors(self, donor_pre: pd.DataFrame) -> None:
        n_clusters = min(self.n_clusters, len(donor_pre))
        self._kmeans = KMeans(
            n_clusters=n_clusters, random_state=self.random_state, n_init=10
        )
        self._kmeans.fit(donor_pre.values)

    # ------------------------------------------------------------------
    # 步骤 2：目标单元匹配最近簇
    # ------------------------------------------------------------------

    def _match_cluster(
        self, treated_pre: pd.Series, donor_pre: pd.DataFrame
    ) -> list[str]:
        assert self._kmeans is not None
        labels = self._kmeans.labels_            # (n_donors,)
        centers = self._kmeans.cluster_centers_  # (K, n_pre)

        dists = np.linalg.norm(centers - treated_pre.values[None, :], axis=1)
        best_cluster = int(np.argmin(dists))

        donor_ids = donor_pre.index[labels == best_cluster].tolist()
        if len(donor_ids) == 0:
            donor_ids = donor_pre.index.tolist()
        return donor_ids

    # ------------------------------------------------------------------
    # 步骤 3：NNLS 合成权重（非负约束 + 归一化，类原始 SC 约束）
    # ------------------------------------------------------------------

    def _fit_nnls(
        self,
        treated_pre: pd.Series,
        donor_pre_subset: pd.DataFrame,
    ) -> np.ndarray:
        X = donor_pre_subset.values.T  # (n_pre, n_donors_subset)
        y = treated_pre.values         # (n_pre,)
        w, _ = nnls(X, y)
        total = w.sum()
        w = w / total if total > 1e-10 else np.ones(len(w)) / len(w)
        return w

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        donors: list[str],
        treated: list[str],
        pre_cols: list[str],
    ) -> "ClusterSC":
        """拟合 ClusterSC 模型。"""
        donor_pre = df.loc[donors, pre_cols]

        # 步骤 1：聚类
        self._cluster_donors(donor_pre)

        for unit in treated:
            treated_pre = df.loc[unit, pre_cols]

            # 步骤 2：匹配最近簇
            sub_donors = self._match_cluster(treated_pre, donor_pre)
            self._donor_ids_per_unit[unit] = sub_donors

            # 步骤 3：NNLS 合成权重
            donor_pre_subset = donor_pre.loc[sub_donors]
            self._weights[unit] = self._fit_nnls(treated_pre, donor_pre_subset)

        return self

    def predict_counterfactual(
        self,
        df: pd.DataFrame,
        treated: list[str],
        post_cols: list[str],
    ) -> pd.DataFrame:
        """预测干预后反事实。"""
        results = {}
        for unit in treated:
            sub_donors = self._donor_ids_per_unit[unit]
            w = self._weights[unit]
            donor_post = df.loc[sub_donors, post_cols].values  # (n_sub, n_post)
            cf = w @ donor_post                                  # (n_post,)
            results[unit] = cf
        return pd.DataFrame(results, index=post_cols).T

    def estimate_att(
        self,
        df: pd.DataFrame,
        treated: list[str],
        post_cols: list[str],
    ) -> pd.DataFrame:
        """估计 ATT = 实际值 - 合成反事实值。"""
        cf = self.predict_counterfactual(df, treated, post_cols)
        rows = []
        for unit in treated:
            for period in post_cols:
                actual = df.loc[unit, period]
                counterfactual = cf.loc[unit, period]
                rows.append({
                    "unit": unit,
                    "period": period,
                    "actual": actual,
                    "counterfactual": counterfactual,
                    "att": actual - counterfactual,
                })
        return pd.DataFrame(rows)

    def pre_treatment_fit(
        self,
        df: pd.DataFrame,
        treated: list[str],
        pre_cols: list[str],
    ) -> pd.DataFrame:
        """干预前 RMSPE，验证合成控制质量。"""
        rows = []
        for unit in treated:
            sub_donors = self._donor_ids_per_unit[unit]
            w = self._weights[unit]
            donor_pre = df.loc[sub_donors, pre_cols].values  # (n_sub, n_pre)
            y_pred = w @ donor_pre                            # (n_pre,)
            y_true = df.loc[unit, pre_cols].values
            rmspe = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
            rows.append({"unit": unit, "rmspe": rmspe})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 自测（self-test）
# ---------------------------------------------------------------------------

def _run_self_test() -> None:
    """内置自测：验证 ClusterSC 的 ATT 估计精度与缩减供体池效果。"""
    print("=" * 60)
    print("ClusterSC Self-Test")
    print("=" * 60)

    TRUE_EFFECT = 15.0

    # 1. 生成低噪声数据（noise=3, individual=5）以保证 NNLS 收敛
    df, donors, treated, pre_cols, post_cols = simulate_geo_data(
        n_donors=200, n_treated=1, n_pre=24, n_post=6,
        treatment_effect=TRUE_EFFECT, seed=42
    )
    print(f"数据: {df.shape}  供体: {len(donors)}  干预前: {len(pre_cols)}  干预后: {len(post_cols)}")

    # 2. 拟合
    model = ClusterSC(n_clusters=10, random_state=42)
    model.fit(df, donors, treated, pre_cols)

    # 3. 干预前 RMSPE
    pre_fit = model.pre_treatment_fit(df, treated, pre_cols)
    rmspe = pre_fit["rmspe"].iloc[0]
    print(f"\n干预前 RMSPE: {rmspe:.4f}  (目标 < 5)")
    assert rmspe < 20.0, f"RMSPE 过高: {rmspe:.4f}"

    # 4. ATT 估计
    att_df = model.estimate_att(df, treated, post_cols)
    avg_att = att_df["att"].mean()
    print(f"\n各期 ATT:\n{att_df[['period', 'actual', 'counterfactual', 'att']].to_string(index=False)}")
    print(f"\n平均 ATT={avg_att:.4f}  真实效应={TRUE_EFFECT}")

    # 5. 容差：±60%（NNLS 在聚类子集上的合理容差范围）
    assert abs(avg_att - TRUE_EFFECT) < TRUE_EFFECT * 0.6, (
        f"ATT 偏差过大: avg_att={avg_att:.4f}, true={TRUE_EFFECT}"
    )

    # 6. 缩减供体池 < 全量
    sub_size = len(model._donor_ids_per_unit[treated[0]])
    print(f"\n缩减供体池: {sub_size}  原始: {len(donors)}")
    assert sub_size < len(donors), "缩减供体池未能减小"

    # 7. 权重和为 1
    w = model._weights[treated[0]]
    assert abs(w.sum() - 1.0) < 1e-6, f"权重和={w.sum()}"

    print("\n[PASS] 所有断言通过，ClusterSC 自测成功。")
    print("=" * 60)


def main() -> None:
    """母婴品牌 TikTok 广告地区级增量归因示例。"""
    print("\n[ClusterSC Demo] 母婴品牌跨国区域广告增量归因")
    print("-" * 50)

    df, donors, treated, pre_cols, post_cols = simulate_geo_data(
        n_donors=500, n_treated=1, n_pre=12, n_post=3,
        treatment_effect=20.0, seed=0,
    )
    model = ClusterSC(n_clusters=15)
    model.fit(df, donors, treated, pre_cols)

    att_df = model.estimate_att(df, treated, post_cols)
    print("处理效应（加州 TikTok 大促月度增量）:")
    print(att_df[["period", "actual", "counterfactual", "att"]].to_string(index=False))
    print(f"平均月度增量: {att_df['att'].mean():.2f}")

    _run_self_test()


if __name__ == "__main__":
    main()
