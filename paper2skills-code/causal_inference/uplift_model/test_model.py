"""Smoke test for uplift_model.

Run: python3 -m pytest test_model.py -v
"""
import numpy as np

from .model import UpliftModel, generate_sample_data


def test_generate_sample_data_shape():
    data = generate_sample_data(n_samples=200)
    assert data is not None
    if isinstance(data, tuple):
        assert len(data) >= 3
    else:
        assert hasattr(data, "shape") or hasattr(data, "columns")


def test_uplift_model_fit_predict_runs():
    np.random.seed(0)
    data = generate_sample_data(n_samples=300)
    if isinstance(data, tuple) and len(data) >= 3:
        X, treatment, outcome = data[0], data[1], data[2]
    else:
        X = data.drop(columns=["treatment", "outcome"]).values
        treatment = data["treatment"].values
        outcome = data["outcome"].values

    model = UpliftModel()
    model.fit(X, treatment, outcome)
    preds = model.predict(X)
    assert preds is not None
    assert len(preds) == len(X)


if __name__ == "__main__":
    test_generate_sample_data_shape()
    test_uplift_model_fit_predict_runs()
    print("OK")
