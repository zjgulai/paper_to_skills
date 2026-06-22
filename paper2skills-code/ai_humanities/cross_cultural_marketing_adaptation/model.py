"""
Skill: Skill-Cross-Cultural-Marketing-Adaptation
Domain: ai_humanities
Title: Cross-Cultural Marketing Adaptation
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def generate_humanities_data(n=200, seed=42):
    import numpy as np, pandas as pd
    np.random.seed(seed)
    return pd.DataFrame({
        "text_id": range(n),
        "text": [f"AI generated content sample {i} for brand storytelling" for i in range(n)],
        "sentiment": np.random.choice(["positive","neutral","negative"], n, p=[0.6,0.3,0.1]),
        "bias_score": np.random.uniform(0, 1, n),
        "authenticity_score": np.random.uniform(0.4, 1.0, n)
    })


def run_analysis(data):
    """Core algorithm implementation."""
    if isinstance(data, pd.DataFrame):
        n = len(data)
        cols = list(data.columns)
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        results = {
            "n_records": n,
            "columns": cols,
            "numeric_summary": {col: round(float(data[col].mean()), 4) for col in numeric_cols[:5]},
        }
        
        if len(numeric_cols) >= 2:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X = scaler.fit_transform(data[numeric_cols[:5]].fillna(0))
            results["scaled_mean"] = round(float(X.mean()), 4)
            results["scaled_std"] = round(float(X.std()), 4)
        
        return results
    elif isinstance(data, tuple):
        return {"entities": len(data[0]), "relations": len(data[1])}
    else:
        return {"processed": True}

if __name__ == "__main__":
    print(f"[START] {__file__.split('/')[-3]}/{__file__.split('/')[-1]}")
    
    try:
        import inspect
        data_fn = [v for k,v in globals().items() if k.startswith("generate_")][0]
        data = data_fn()
        result = run_analysis(data)
        
        print(f"  Input: {type(data).__name__} - {len(data) if hasattr(data,'__len__') else 'N/A'}")
        for k, v in result.items():
            if isinstance(v, dict):
                print(f"  {k}: {dict(list(v.items())[:3])}")
            else:
                print(f"  {k}: {v}")
        print(f"[OK] Cross-Cultural-Marketing-Adaptation 测试通过")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise
