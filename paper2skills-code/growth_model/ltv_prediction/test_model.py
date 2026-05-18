"""Smoke test for ltv_prediction (data generation only — full DNN training is too slow for smoke test)."""
from .model import generate_ltv_data


def test_generate_ltv_data():
    df = generate_ltv_data(n_samples=200)
    assert df is not None
    assert len(df) == 200


if __name__ == "__main__":
    test_generate_ltv_data()
    print("OK")
