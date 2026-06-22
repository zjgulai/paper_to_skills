"""DML Cohort CATE skeleton based on arXiv:2409.02332 (Amazon, ECML PKDD 2023).

Uses EconML if available. The Amazon paper-specific Spark distribution layer
is omitted (econml on a single node handles millions of rows).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

try:
    from econml.dml import LinearDML
    HAS_ECONML = True
except ImportError:
    HAS_ECONML = False


def simulate_baby_ecom_data(n: int = 5000, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 50))
    baby_age_months = rng.integers(0, 24, n)
    X[:, 0] = baby_age_months / 24.0

    propensity = 1.0 / (1.0 + np.exp(-0.3 * X[:, 0] - 0.5 * X[:, 1]))
    D = (rng.random(n) < propensity).astype(int)

    true_cate = 50.0 + 30.0 * (1 - baby_age_months / 24.0)
    Y = 200.0 + true_cate * D + 20.0 * X[:, 0] + rng.standard_normal(n) * 50.0
    return X, D, Y, baby_age_months, true_cate


def build_cohort_features(X: np.ndarray, n_components: int = 10, n_clusters: int = 5) -> np.ndarray:
    pca = PCA(n_components=n_components).fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(pca)
    dists = kmeans.transform(pca)
    psi = 1.0 / (dists + 1e-8)
    psi = psi / psi.sum(axis=1, keepdims=True)
    return psi


def fit_dml_cohort_cate(X: np.ndarray, D: np.ndarray, Y: np.ndarray, psi: np.ndarray):
    if not HAS_ECONML:
        raise ImportError("Install econml: pip install econml")
    model = LinearDML(
        model_y=GradientBoostingRegressor(n_estimators=100),
        model_t=GradientBoostingRegressor(n_estimators=100),
        featurizer=None,
        cv=3,
        random_state=42,
    )
    model.fit(Y, D, X=psi, W=X)
    return model


def main() -> None:
    X, D, Y, ages, true_cate = simulate_baby_ecom_data()
    psi = build_cohort_features(X)
    print(f"Cohort feature shape: {psi.shape}")

    if not HAS_ECONML:
        print("econml 未安装, 跳过 DML 拟合(生产部署需要 pip install econml)")
        return

    model = fit_dml_cohort_cate(X, D, Y, psi)
    cate_hat = model.effect(psi)

    df = pd.DataFrame({"age_months": ages, "cate_hat": cate_hat, "true_cate": true_cate})
    df["cohort"] = pd.cut(df["age_months"], bins=[0, 3, 6, 12, 24], labels=["0-3月", "4-6月", "7-12月", "13-24月"])
    summary = df.groupby("cohort", observed=False).agg(estimated=("cate_hat", "mean"), true=("true_cate", "mean")).round(2)
    print(summary)


if __name__ == "__main__":
    main()
