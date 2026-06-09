"""Smoke test for causal_forest."""
from .model import CausalForestAttribution, generate_multimarket_data


def test_causal_forest_smoke():
    X, treatment, outcome, market = generate_multimarket_data(n_samples=400)
    assert X is not None
    assert len(X) > 0
    assert len(treatment) == len(outcome) == len(market) == len(X)

    model = CausalForestAttribution()
    model.fit(X.values, treatment, outcome)
    preds = model.predict(X.values)
    assert len(preds) == len(X)


if __name__ == "__main__":
    test_causal_forest_smoke()
    print("OK")
