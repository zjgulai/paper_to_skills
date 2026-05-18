"""Smoke test for causal_forest."""
from .model import CausalForestAttribution, generate_multimarket_data


def test_causal_forest_smoke():
    df = generate_multimarket_data(n_samples=400)
    assert df is not None
    assert len(df) > 0

    model = CausalForestAttribution()
    feature_cols = [c for c in df.columns if c not in ('treatment', 'outcome', 'market')]
    model.fit(df[feature_cols].values, df['treatment'].values, df['outcome'].values)
    preds = model.predict(df[feature_cols].values)
    assert len(preds) == len(df)


if __name__ == "__main__":
    test_causal_forest_smoke()
    print("OK")
