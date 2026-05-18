"""Smoke test for demand_forecasting."""
from .model import DemandForecaster, generate_sample_data


def test_forecaster_runs():
    df = generate_sample_data(n_weeks=52)
    assert df is not None
    assert len(df) > 0

    forecaster = DemandForecaster()
    forecaster.fit(df)
    forecast = forecaster.predict(periods=4)
    assert forecast is not None


if __name__ == "__main__":
    test_forecaster_runs()
    print("OK")
