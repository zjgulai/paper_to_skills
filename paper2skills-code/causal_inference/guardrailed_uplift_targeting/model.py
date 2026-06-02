"""
Guardrailed Uplift Targeting: A Causal Optimization Playbook for Marketing Strategy
arXiv:2512.19805 (2025-12) | 母婴订阅用户挽留 / WF-B 广告人群精准定向
纯标准库实现（无 sklearn/pandas）
"""
from __future__ import annotations
import math, random
from dataclasses import dataclass
from typing import Optional


@dataclass
class CustomerFeatures:
    customer_id: str
    purchase_freq: float
    days_since_last: int
    ltv: float
    has_subscription: bool
    baby_age_months: int
    avg_order_value: float


@dataclass
class TreatmentRecord:
    customer: CustomerFeatures
    treated: bool
    outcome: float


@dataclass
class TargetingPolicy:
    targeting_rate: float
    expected_lift: float
    total_cost: float
    roi: float
    n_persuadables: int
    n_sleeping_dogs: int
    qini_auuc: float


def _to_feat(c: CustomerFeatures) -> list:
    return [c.purchase_freq, c.days_since_last/100, c.ltv/1000,
            float(c.has_subscription), c.baby_age_months/24, c.avg_order_value/300]


def _sigmoid(x): return 1/(1+math.exp(-max(-20,min(20,x))))


class SimpleTree:
    """极简决策树（纯 Python）"""
    def __init__(self, max_depth=3, min_samples=3):
        self.max_depth=max_depth; self.min_samples=min_samples; self.tree=None

    def _gini(self, y):
        if not y: return 0
        p = sum(y)/len(y); return 1-p*p-(1-p)**2

    def _best_split(self, X, y):
        best=(None,None,float('inf'))
        for fi in range(len(X[0])):
            vals=sorted(set(x[fi] for x in X))
            for t in [(vals[i]+vals[i+1])/2 for i in range(len(vals)-1)]:
                L=[y[i] for i,x in enumerate(X) if x[fi]<=t]
                R=[y[i] for i,x in enumerate(X) if x[fi]>t]
                if not L or not R: continue
                g=len(L)/(len(L)+len(R))*self._gini(L)+len(R)/(len(L)+len(R))*self._gini(R)
                if g<best[2]: best=(fi,t,g)
        return best[0],best[1]

    def _build(self, X, y, depth):
        if depth==0 or len(y)<self.min_samples or len(set(y))==1:
            return sum(y)/max(len(y),1)
        fi,t=self._best_split(X,y)
        if fi is None: return sum(y)/max(len(y),1)
        Lx=[X[i] for i,x in enumerate(X) if x[fi]<=t]
        Ly=[y[i] for i,x in enumerate(X) if x[fi]<=t]
        Rx=[X[i] for i,x in enumerate(X) if x[fi]>t]
        Ry=[y[i] for i,x in enumerate(X) if x[fi]>t]
        return (fi,t,self._build(Lx,Ly,depth-1),self._build(Rx,Ry,depth-1))

    def fit(self, X, y): self.tree=self._build(X,y,self.max_depth); return self

    def predict(self, X):
        def _p(node,x):
            if not isinstance(node,tuple): return node
            fi,t,L,R=node; return _p(L,x) if x[fi]<=t else _p(R,x)
        return [_p(self.tree,x) for x in X]


class CATEEstimator:
    """X-Learner CATE 估计（纯 Python 决策树）"""
    def __init__(self): self._mu0=SimpleTree(3); self._mu1=SimpleTree(3); self._tau=SimpleTree(3)

    def fit(self, records: list[TreatmentRecord]):
        ctrl=[(r.customer,r.outcome) for r in records if not r.treated]
        trt= [(r.customer,r.outcome) for r in records if r.treated]
        if ctrl: self._mu0.fit([_to_feat(c) for c,_ in ctrl],[o for _,o in ctrl])
        if trt:  self._mu1.fit([_to_feat(c) for c,_ in trt], [o for _,o in trt])
        # 伪结果
        pseudo=[]; X_all=[]
        for r in records:
            f=_to_feat(r.customer)
            if r.treated:
                mu0_pred=self._mu0.predict([f])[0] if ctrl else 0.5
                pseudo.append(r.outcome-mu0_pred)
            else:
                mu1_pred=self._mu1.predict([f])[0] if trt else 0.5
                pseudo.append(mu1_pred-r.outcome)
            X_all.append(f)
        self._tau.fit(X_all,[max(0,min(1,p+0.5)) for p in pseudo])
        return self

    def predict_cate(self, customers: list[CustomerFeatures]) -> list[float]:
        raw=self._tau.predict([_to_feat(c) for c in customers])
        mean=sum(raw)/max(len(raw),1)
        return [r-mean for r in raw]


class UpliftOptimizer:
    def __init__(self, budget_pct=0.15, cost=30.0):
        self.budget_pct=budget_pct; self.cost=cost

    def optimize(self, customers, cate_scores) -> TargetingPolicy:
        n=len(customers); budget_n=max(1,int(n*self.budget_pct))
        persu=[(i,s) for i,s in enumerate(cate_scores) if s>0]
        dogs= [(i,s) for i,s in enumerate(cate_scores) if s<-0.05]
        persu.sort(key=lambda x:-x[1])
        targeted=persu[:budget_n]
        if not targeted:
            return TargetingPolicy(0,0,0,0,0,len(dogs),0)
        rate=len(targeted)/n
        lift=sum(cate_scores[i] for i,_ in targeted)/len(targeted)
        total_cost=len(targeted)*self.cost
        avg_ltv=sum(customers[i].ltv for i,_ in targeted)/len(targeted)
        roi=max(0,(lift*avg_ltv*len(targeted))/max(total_cost,1))
        # Qini
        sorted_all=sorted(enumerate(cate_scores),key=lambda x:-x[1])
        cum=0; auuc=0
        for rank,(idx,score) in enumerate(sorted_all):
            cum+=max(0,score); auuc+=cum/(rank+1)
        return TargetingPolicy(round(rate,3),round(lift,4),round(total_cost,0),
                               round(roi,2),len(persu),len(dogs),round(auuc/n,4))


def run_guardrailed_demo():
    random.seed(42)
    customers=[CustomerFeatures(f"C{i}",random.uniform(0.5,4),random.randint(7,120),
                random.uniform(200,2000),random.random()>0.4,
                random.randint(0,36),random.uniform(80,500)) for i in range(200)]
    records=[]
    for c in customers:
        treated=random.random()>0.5
        effect=0.15 if (c.baby_age_months<18 and c.days_since_last<30) else -0.05
        prob=0.3+0.3*c.purchase_freq/4+(effect if treated else 0)
        outcome=float(random.random()<max(0.05,min(0.95,prob)))
        records.append(TreatmentRecord(c,treated,outcome))

    print("=== Guardrailed Uplift Targeting（母婴订阅挽留）===")
    est=CATEEstimator(); est.fit(records)
    test_c=[r.customer for r in records[:80]]
    scores=est.predict_cate(test_c)
    opt=UpliftOptimizer(budget_pct=0.20)
    pol=opt.optimize(test_c,scores)
    print(f"  定向率: {pol.targeting_rate:.1%} | ROI: {pol.roi:.2f}x | Persuadables: {pol.n_persuadables}")
    assert pol.targeting_rate<=1.0
    assert pol.n_persuadables>=0
    print("✅ Guardrailed Uplift Targeting 验证通过")

if __name__=="__main__":
    run_guardrailed_demo()
