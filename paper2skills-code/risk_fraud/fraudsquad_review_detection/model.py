"""
FraudSquad — LLM 生成虚假评论检测：LM 嵌入 + 门控图变换器
arXiv:2510.01801 (2025-10) | 母婴 Amazon 刷评检测 / WF-E Review 清洗
纯标准库实现（无 sklearn/pandas）
"""
from __future__ import annotations
import math, random, re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Review:
    review_id: str; user_id: str; product_id: str
    text: str; rating: int; timestamp: int
    is_spam: Optional[bool] = None


@dataclass
class DetectionResult:
    review_id: str; spam_probability: float
    is_spam_predicted: bool; confidence: str
    signals: list[str]


class TFIDFEmbedder:
    """轻量 TF-IDF（生产替换为 sentence-transformers）"""
    def __init__(self, max_f=30): self.max_f=max_f; self.vocab={}; self.idf={}

    def _tok(self,t): return re.findall(r"[a-zA-Z一-鿿]+",t.lower())

    def fit(self, texts):
        n=len(texts); df=defaultdict(int)
        for t in texts:
            for w in set(self._tok(t)): df[w]+=1
        idf_s={w:math.log((n+1)/(c+1))+1 for w,c in df.items()}
        top=sorted(idf_s,key=idf_s.get,reverse=True)[:self.max_f]
        self.vocab={w:i for i,w in enumerate(top)}
        self.idf={w:idf_s[w] for w in top}
        return self

    def transform(self, texts):
        res=[]
        for t in texts:
            toks=self._tok(t); tf=defaultdict(float)
            for w in toks: tf[w]+=1
            n=max(len(toks),1); v=[0.0]*self.max_f
            for w,i in self.vocab.items():
                if w in tf: v[i]=(tf[w]/n)*self.idf[w]
            norm=math.sqrt(sum(x*x for x in v))+1e-9
            res.append([x/norm for x in v])
        return res


class ReviewGraph:
    def __init__(self):
        self.nodes={}; self.user_rev=defaultdict(list); self.prod_rev=defaultdict(list)

    def add(self,r: Review):
        self.nodes[r.review_id]=r
        self.user_rev[r.user_id].append(r.review_id)
        self.prod_rev[r.product_id].append(r.review_id)

    def neighbors(self,rid):
        r=self.nodes[rid]
        return list(set(self.user_rev[r.user_id]+self.prod_rev[r.product_id])-{rid})


def graph_features(r: Review, g: ReviewGraph) -> list:
    u_revs=[g.nodes[i] for i in g.user_rev[r.user_id] if i in g.nodes]
    p_revs=[g.nodes[i] for i in g.prod_rev[r.product_id] if i in g.nodes]
    n_u=len(u_revs)
    avg_r=sum(x.rating for x in u_revs)/max(n_u,1)
    five=sum(1 for x in u_revs if x.rating==5)/max(n_u,1)
    ts=[x.timestamp for x in u_revs]
    if len(ts)>1:
        m=sum(ts)/len(ts); std=math.sqrt(sum((t-m)**2 for t in ts)/len(ts))
        conc=1/(std+1)
    else: conc=0.5
    nb=[g.nodes[i] for i in g.neighbors(r.review_id) if i in g.nodes]
    nb_cons=sum(1 for x in nb if abs(x.rating-r.rating)<=1)/max(len(nb),1) if nb else 0.5
    p_avg=sum(x.rating for x in p_revs)/max(len(p_revs),1)
    return [min(n_u/20,1),avg_r/5,five,conc,nb_cons,abs(r.rating-p_avg)/4]


class FraudSquadDetector:
    def __init__(self, max_f=20):
        self.emb=TFIDFEmbedder(max_f); self.graph=None
        self._spam_mean=0.7; self._legit_mean=0.4

    def _gate(self, h_self, h_nb, W):
        dim=len(h_self)
        concat=h_self+h_nb
        gate=[1/(1+math.exp(-max(-10,min(10,sum(W[i][j]*concat[j] for j in range(len(concat)))))))
              for i in range(dim)]
        return [gate[i]*h_nb[i]+(1-gate[i])*h_self[i] for i in range(dim)]

    def fit(self, reviews, graph, labeled_ids):
        self.graph=graph
        random.seed(42)
        texts=[r.text for r in reviews]
        self.emb.fit(texts)
        t_feats={r.review_id:e for r,e in zip(reviews,self.emb.transform(texts))}
        # 融合图特征
        node_feats={}
        for r in reviews:
            gf=graph_features(r,graph)
            tf=t_feats[r.review_id][:6]
            node_feats[r.review_id]=[a+b for a,b in zip(tf,gf)]
        dim=len(next(iter(node_feats.values())))
        W=[[random.gauss(0,0.1) for _ in range(dim*2)] for _ in range(dim)]
        # 一轮消息传播
        updated={k:v[:] for k,v in node_feats.items()}
        for rid,h in updated.items():
            nb=[updated[n] for n in graph.neighbors(rid) if n in updated]
            if nb:
                h_nb=[sum(v[i] for v in nb)/len(nb) for i in range(dim)]
                updated[rid]=self._gate(h,h_nb,W)
        self._feat=updated
        # 从标注样本学习阈值
        spam_n=[math.sqrt(sum(v**2 for v in updated[r])) for r in labeled_ids
                if r in updated and graph.nodes[r].is_spam]
        legit_n=[math.sqrt(sum(v**2 for v in updated[r])) for r in labeled_ids
                 if r in updated and graph.nodes[r].is_spam is False]
        if spam_n: self._spam_mean=sum(spam_n)/len(spam_n)
        if legit_n: self._legit_mean=sum(legit_n)/len(legit_n)
        return self

    def detect(self, review_ids, threshold=0.5):
        results=[]
        for rid in review_ids:
            if rid not in self._feat: continue
            norm=math.sqrt(sum(v**2 for v in self._feat[rid]))
            span=abs(self._spam_mean-self._legit_mean)+1e-9
            prob=min(0.99,max(0.01,abs(norm-self._legit_mean)/(span+abs(norm-self._legit_mean))))
            r=self.graph.nodes.get(rid); signals=[]
            if r:
                n_u=len(self.graph.user_rev.get(r.user_id,[]))
                if n_u>3: signals.append(f"高频用户({n_u}条)")
                if r.rating==5: signals.append("满分评价")
                if len(r.text)<20: signals.append("文本过短")
            conf="HIGH" if abs(prob-0.5)>0.3 else ("MEDIUM" if abs(prob-0.5)>0.15 else "LOW")
            results.append(DetectionResult(rid,round(prob,3),prob>=threshold,conf,signals))
        return results


def run_fraudsquad_demo():
    g=ReviewGraph(); reviews=[]
    genuine_texts=["Great formula, baby loves it","Bottle leaks slightly but ok",
                   "Fast shipping, intact packaging","Matches description well","Good price for organic"]
    spam_texts=["Excellent product highly recommended best","Perfect wonderful amazing superb outstanding",
                "Best formula ever five stars must buy","Amazing quality perfect best price buy now",
                "Excellent excellent great product love it"]
    for i,t in enumerate(genuine_texts):
        r=Review(f"R{i}",f"U_real_{i}","P001",t,random.randint(3,5),i*7,is_spam=False)
        reviews.append(r); g.add(r)
    for i,t in enumerate(spam_texts):
        r=Review(f"R{i+10}","U_spammer","P001",t,5,i,is_spam=True)
        reviews.append(r); g.add(r)
    labeled=[r.review_id for r in reviews]
    new=[Review("N1","U_real_0","P001","Good product baby happy",4,50),
         Review("N2","U_spammer","P001","Best amazing five stars buy now",5,51),
         Review("N3","U_new","P001","Packaging damaged but ok inside",3,52)]
    for r in new: reviews.append(r); g.add(r)
    det=FraudSquadDetector(max_f=15); det.fit(reviews,g,labeled)
    results=det.detect([r.review_id for r in new])
    print("=== FraudSquad 虚假评论检测（母婴 Amazon）===")
    for res,rev in zip(results,new):
        tag="🚨 虚假" if res.is_spam_predicted else "✅ 真实"
        print(f"  {tag} [{res.confidence}] {rev.review_id}: p={res.spam_probability:.2f}")
    print("✅ FraudSquad 检测演示完成")

if __name__=="__main__":
    random.seed(42); run_fraudsquad_demo()
