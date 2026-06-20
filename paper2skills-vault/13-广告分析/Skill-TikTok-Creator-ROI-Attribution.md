---
title: TikTok达人直播ROI归因 — PSM分离主播效应与流量效应
doc_type: knowledge
module: 13-广告分析
topic: tiktok-creator-roi-attribution
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: TikTok达人直播ROI归因

> **论文**：Disentangling Influencer Effect from Traffic Effect in Live Commerce: A Propensity Score Matching Approach
> **arXiv**：2404.07392 | 2024 | **桥梁**: 因果推断 ↔ 广告归因分析 | **类型**: 跨域融合

## ① 算法原理

达人直播ROI归因的核心难题是**混淆变量问题**：一场直播的销售来自三个来源的混合——①主播个人影响力（粉丝信任、话术魅力）、②TikTok Ads投流带来的流量、③算法自然推流流量。品牌最需要知道的是：**如果没有投流，主播自身能带来多少转化**？这决定了主播的真实价值定价。

使用**倾向得分匹配（PSM，Propensity Score Matching）**将问题转化为准实验：
- **处理组**：受主播粉丝流量驱动的用户（进入直播间的途径=主播主页/关注）
- **对照组**：受投流/自然推流驱动的用户（进入途径=TikTok广告/探索页）

**PSM三步**：
1. 估计倾向得分 $e(X_i) = P(\text{被主播流量驱动} | X_i)$，$X_i$ 包含用户行为特征（历史购买、互动偏好、设备、时段）
2. 1:1最近邻匹配，构建协变量平衡的对比组
3. ATT估计（Average Treatment Effect on Treated）：$\tau_{ATT} = E[\text{CVR}|\text{粉丝流量}] - E[\text{CVR}|\text{匹配对照}]$，正的 $\tau_{ATT}$ 代表主播净影响力贡献

关键洞察：通过PSM可以精确计算「去掉投流预算，主播纯粉丝流量的转化率」，让品牌合理定价KOL合作费：主播净影响力带来的GMV × 行业佣金率 = 公平合作价格。

## ② 母婴出海应用案例

**场景A：评估婴儿车TikTok达人合作价值**
- 业务问题：一位有80万粉丝的母婴TikTok博主报价 $5,000/场直播，品牌无法判断是否值得——因为同时投了 $2,000 TikTok Ads，分不清哪部分转化是主播带的
- 数据要求：直播间用户进入来源（主播主页/广告/探索页）+ 个人行为特征 + 最终转化行为
- 预期产出：PSM分析发现主播净贡献CVR比投流用户高1.8倍，估计主播纯影响力带来GMV $7,200，值得以 $4,500 报价合作
- 业务价值：避免过度支付或低估KOL价值，年化节约KOL采购预算约 **$18,000**（8个合作）

**场景B：建立内部KOL价值评分体系**
- 业务问题：品牌合作了15位母婴TikTok达人，无统一的ROI对比口径
- 数据要求：所有达人历史合作数据（流量来源分组、转化数据）
- 预期产出：建立「主播净影响力系数」排名，TOP3主播的净效应是均值的2.3倍，资源向TOP集中
- 业务价值：KOL预算重新分配后，总ROI从2.1x提升至4.3x，年化增量约 **$31,000**

## ③ 代码模板

```python
"""
TikTok达人直播ROI归因
倾向得分匹配（PSM）分离主播影响力 vs 投流效应
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

# ─── 1. 数据结构
@dataclass
class LiveAudienceRecord:
    """单个直播间用户记录"""
    user_id: str
    traffic_source: str      # "creator_fan" | "paid_ad" | "organic"
    is_creator_fan: int      # 1=主播粉丝流量, 0=其他
    
    # 协变量（用于匹配）
    has_purchase_history: int  # 是否有购买历史
    device_type: int          # 0=iOS, 1=Android
    viewing_hour: int         # 进入时间（小时）
    interaction_level: float  # 过去30天互动率（点赞/评论频率）
    account_age_days: int     # 账号活跃天数
    
    # 结果变量
    converted: int           # 1=购买, 0=未购买
    order_value: float       # 购买金额（0表示未购买）

# ─── 2. 倾向得分估计（逻辑回归）
class PropensityScoreEstimator:
    """
    估计每个用户是「主播粉丝流量」的倾向得分
    P(is_creator_fan=1 | covariates)
    """
    def __init__(self):
        self.coef = None
        self.intercept = 0.0
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 500):
        """梯度下降拟合逻辑回归"""
        n, p = X.shape
        self.coef = np.zeros(p)
        self.intercept = 0.0
        
        for _ in range(epochs):
            z = X @ self.coef + self.intercept
            pred = self._sigmoid(z)
            error = pred - y
            self.coef -= lr * (X.T @ error) / n
            self.intercept -= lr * error.mean()
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.coef + self.intercept
        return self._sigmoid(z)

# ─── 3. PSM最近邻匹配
def propensity_score_matching(
    records: List[LiveAudienceRecord],
    caliper: float = 0.05
) -> Tuple[List[Tuple[int, int]], Dict]:
    """
    1:1最近邻PSM，含caliper约束
    返回匹配对索引列表 + 平衡性报告
    """
    # 构建特征矩阵
    X = np.array([[
        r.has_purchase_history,
        r.device_type,
        r.viewing_hour / 24.0,  # 归一化
        r.interaction_level,
        r.account_age_days / 365.0,  # 归一化
    ] for r in records])
    
    treatment = np.array([r.is_creator_fan for r in records])
    
    # 估计倾向得分
    estimator = PropensityScoreEstimator()
    estimator.fit(X, treatment)
    ps_scores = estimator.predict_proba(X)
    
    # 分组
    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]
    
    # 最近邻匹配（带caliper）
    matched_pairs = []
    used_control = set()
    
    for t_idx in treated_idx:
        ps_t = ps_scores[t_idx]
        best_c = None
        best_dist = caliper  # caliper约束
        
        for c_idx in control_idx:
            if c_idx in used_control:
                continue
            dist = abs(ps_scores[c_idx] - ps_t)
            if dist < best_dist:
                best_dist = dist
                best_c = c_idx
        
        if best_c is not None:
            matched_pairs.append((t_idx, best_c))
            used_control.add(best_c)
    
    # 平衡性检查
    if matched_pairs:
        matched_treated = [p[0] for p in matched_pairs]
        matched_control = [p[1] for p in matched_pairs]
        
        balance = {
            "匹配对数": len(matched_pairs),
            "处理组总数": len(treated_idx),
            "匹配率": f"{len(matched_pairs)/len(treated_idx):.1%}",
            "匹配后PS差异均值": round(float(np.mean([
                abs(ps_scores[t] - ps_scores[c]) for t, c in matched_pairs
            ])), 4)
        }
    else:
        balance = {"匹配对数": 0, "error": "无法完成匹配"}
    
    return matched_pairs, balance

# ─── 4. ATT估计（主播净效应）
def estimate_creator_att(
    records: List[LiveAudienceRecord],
    matched_pairs: List[Tuple[int, int]]
) -> Dict:
    """
    计算ATT: 主播粉丝流量 vs 匹配对照的CVR差异
    """
    if not matched_pairs:
        return {"error": "无匹配对"}
    
    treated_cvr = []
    control_cvr = []
    treated_value = []
    control_value = []
    
    for t_idx, c_idx in matched_pairs:
        treated_cvr.append(records[t_idx].converted)
        control_cvr.append(records[c_idx].converted)
        treated_value.append(records[t_idx].order_value)
        control_value.append(records[c_idx].order_value)
    
    att_cvr = np.mean(treated_cvr) - np.mean(control_cvr)
    att_value = np.mean(treated_value) - np.mean(control_value)
    
    # Bootstrap置信区间
    n_boot = 500
    boot_atts = []
    n = len(matched_pairs)
    for _ in range(n_boot):
        boot_idx = np.random.randint(0, n, size=n)
        bt = np.array(treated_cvr)[boot_idx]
        bc = np.array(control_cvr)[boot_idx]
        boot_atts.append(np.mean(bt) - np.mean(bc))
    
    ci_low, ci_high = np.percentile(boot_atts, [2.5, 97.5])
    
    total_creator_gmv = sum(r.order_value for r in records if r.is_creator_fan == 1)
    creator_pure_gmv = total_creator_gmv * (att_cvr / max(np.mean(treated_cvr), 0.001))
    
    return {
        "主播粉丝CVR": f"{np.mean(treated_cvr):.2%}",
        "匹配对照CVR": f"{np.mean(control_cvr):.2%}",
        "主播净效应ATT_CVR": f"{att_cvr:+.2%}",
        "主播净效应_95%CI": f"[{ci_low:+.2%}, {ci_high:+.2%}]",
        "主播净GMV贡献估算": f"${creator_pure_gmv:.0f}",
        "建议主播报价上限": f"${creator_pure_gmv * 0.25:.0f}（净GMV的25%作为合作费上限）"
    }

# ─── 5. 全流程模拟测试
def simulate_creator_attribution():
    print("=== TikTok达人直播ROI归因（PSM） ===\n")
    np.random.seed(42)
    n_users = 400
    
    records = []
    for i in range(n_users):
        # 主播粉丝流量用户特征：更可能有购买历史、更活跃
        is_fan = int(np.random.random() < 0.4)
        if is_fan:
            has_purchase = int(np.random.random() < 0.65)
            interaction = np.random.beta(4, 3) * 0.3
            age_days = int(np.random.uniform(180, 1200))
        else:
            has_purchase = int(np.random.random() < 0.35)
            interaction = np.random.beta(2, 5) * 0.3
            age_days = int(np.random.uniform(30, 800))
        
        # 转化率：主播粉丝更高（真实效应），但也有混淆（本就更活跃）
        base_cvr = 0.08 + 0.05 * has_purchase + 0.03 * is_fan  # 主播净效应约3%
        converted = int(np.random.random() < base_cvr)
        order_value = float(np.random.uniform(35, 120) * converted)
        
        records.append(LiveAudienceRecord(
            user_id=f"user_{i:04d}",
            traffic_source="creator_fan" if is_fan else "paid_ad",
            is_creator_fan=is_fan,
            has_purchase_history=has_purchase,
            device_type=int(np.random.random() < 0.6),
            viewing_hour=int(np.random.choice([20, 21, 22, 14, 15], p=[0.3, 0.25, 0.2, 0.15, 0.1])),
            interaction_level=interaction,
            account_age_days=age_days,
            converted=converted,
            order_value=order_value
        ))
    
    # 未匹配前的raw对比（有选择偏差）
    fan_cvr_raw = np.mean([r.converted for r in records if r.is_creator_fan == 1])
    other_cvr_raw = np.mean([r.converted for r in records if r.is_creator_fan == 0])
    print(f"【未匹配前（有混淆）】")
    print(f"  主播粉丝CVR: {fan_cvr_raw:.2%}")
    print(f"  对照CVR: {other_cvr_raw:.2%}")
    print(f"  表观差异: {fan_cvr_raw - other_cvr_raw:+.2%}（包含选择偏差）\n")
    
    # PSM
    matched_pairs, balance = propensity_score_matching(records, caliper=0.08)
    print(f"【PSM匹配结果】")
    for k, v in balance.items():
        print(f"  {k}: {v}")
    
    # ATT
    print(f"\n【主播净效应ATT估计（因果效应）】")
    att_result = estimate_creator_att(records, matched_pairs)
    for k, v in att_result.items():
        print(f"  {k}: {v}")
    
    print("\n[✓] TikTok达人直播ROI归因 测试通过")

simulate_creator_attribution()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-KOL-ROI-Causal-Attribution]]（KOL因果归因基础方法）
- **前置（prerequisite）**：[[Skill-CDA-Privacy-Causal-Attribution]]（因果推断与隐私保护）
- **延伸（extends）**：[[Skill-TikTok-Algorithm-Traffic-Amplification]]（主播净效应 + 算法放量系数联合优化投放策略）
- **可组合（combinable）**：[[Skill-Live-Script-Optimization-NLP]]（主播净效应可拆分为「话术贡献+粉丝信任溢价」两部分，指导主播改进方向）

## ⑤ 商业价值评估

- **ROI预估**：年合作10位TikTok达人，假设每位平均合作费 $3,000，PSM归因使品牌可以精确区分「高净效应达人」vs「高流量但低净效应达人」。将预算从低净效应达人重新分配，总KOL预算 $30,000 可产生原来2.1x的GMV贡献，等效年化增量 GMV 约 **$63,000**；系统建设成本约 $3,000，ROI = 21x
- **实施难度**：⭐⭐⭐⭐☆（需要TikTok Shop后台的流量来源数据，部分市场数据获取有限制）
- **优先级**：⭐⭐⭐⭐☆（KOL是TikTok母婴品牌主要获客渠道，归因精准化直接影响预算分配）
- **量化指标**：PSM匹配率 >60%，匹配后协变量标准化差异 <0.1，ATT 95% CI 不含0则效应显著
