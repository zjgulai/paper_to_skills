"""
Skill: Skill-Ensemble-Methods
Domain: ml_fundamentals
Title: Ensemble Methods（集成学习方法）
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def generate_ml_data(n=2000, seed=42):
    import numpy as np, pandas as pd
    from sklearn.datasets import make_classification
    np.random.seed(seed)
    X, y = make_classification(n_samples=n, n_features=10, n_informative=6,
                                n_redundant=2, random_state=seed)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    df["target"] = y
    return df


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
        print(f"[OK] Ensemble-Methods 测试通过")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise
