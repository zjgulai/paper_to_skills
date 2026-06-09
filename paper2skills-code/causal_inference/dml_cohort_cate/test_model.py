"""Smoke test for dml_cohort_cate."""
from .model import build_cohort_features, simulate_baby_ecom_data


def test_simulate_returns_consistent_shapes():
    X, D, Y, ages, cate = simulate_baby_ecom_data(n=200)
    assert X.shape == (200, 50)
    assert D.shape == (200,)
    assert Y.shape == (200,)
    assert ages.shape == (200,)
    assert cate.shape == (200,)


def test_cohort_features_sum_to_one_per_row():
    X, _, _, _, _ = simulate_baby_ecom_data(n=100)
    psi = build_cohort_features(X, n_components=5, n_clusters=3)
    assert psi.shape == (100, 3)
    row_sums = psi.sum(axis=1)
    assert all(abs(s - 1.0) < 1e-5 for s in row_sums)


if __name__ == "__main__":
    test_simulate_returns_consistent_shapes()
    test_cohort_features_sum_to_one_per_row()
    print("OK")
