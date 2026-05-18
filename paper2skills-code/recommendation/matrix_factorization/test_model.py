"""Smoke test for matrix_factorization."""
from .model import MatrixFactorization, generate_sample_data


def test_mf_fit_predict():
    data = generate_sample_data()
    mf = MatrixFactorization(n_factors=4, n_iterations=3)
    if isinstance(data, tuple):
        ratings = data[0]
    else:
        ratings = data
    mf.fit(ratings)
    pred = mf.predict(0, 0)
    assert pred is not None


if __name__ == "__main__":
    test_mf_fit_predict()
    print("OK")
