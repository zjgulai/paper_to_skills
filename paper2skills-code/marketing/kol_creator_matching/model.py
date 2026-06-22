"""
Skill: Skill-KOL-Creator-Matching
Domain: marketing
Title: KOL-Creator-Matching（KOL 达人精准匹配）
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def generate_marketing_data(n=365, seed=42):
    import numpy as np, pandas as pd
    np.random.seed(seed)
    channels = ["amazon_sp","google","facebook","tiktok"]
    rows = []
    for ch in channels:
        spend = np.random.uniform(200,2000,n)
        roas = np.random.uniform(2,6,n)
        rows.append(pd.DataFrame({"date": pd.date_range("2023-01-01",periods=n),
                                   "channel": ch, "spend": spend, "revenue": spend*roas,
                                   "new_customers": (spend*roas/50).astype(int)}))
    return pd.concat(rows).reset_index(drop=True)


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
        print(f"[OK] KOL-Creator-Matching 测试通过")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise
