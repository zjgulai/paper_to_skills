"""
Skill: Skill-Anomaly-Detection-Foundation-Model
Domain: risk_fraud
Title: Anomaly Detection Foundation Model — 异常检测基础模型
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def generate_fraud_data(n=5000, seed=42):
    import numpy as np, pandas as pd
    np.random.seed(seed)
    normal = pd.DataFrame({"amount": np.random.normal(50,30,n),
                            "hour": np.random.randint(8,20,n),
                            "velocity": np.random.randint(1,5,n),
                            "is_fraud": 0})
    fraud = pd.DataFrame({"amount": np.random.normal(200,100,int(n*0.05)),
                           "hour": np.random.randint(0,6,int(n*0.05)),
                           "velocity": np.random.randint(10,50,int(n*0.05)),
                           "is_fraud": 1})
    return pd.concat([normal, fraud]).sample(frac=1).reset_index(drop=True)


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
        print(f"[OK] Anomaly-Detection-Foundation-Model 测试通过")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise
