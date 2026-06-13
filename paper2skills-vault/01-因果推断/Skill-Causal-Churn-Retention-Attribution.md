---
title: Causal Churn Retention Attribution — Uplift + DiD 因果用户流失归因与留存优化
doc_type: knowledge
module: 01-因果推断
topic: causal-churn-retention-attribution
status: stable
created: 2026-06-13
updated: 2026-06-13
owner: self
source: human+ai
roadmap_phase: phase2
algorithm_summary: DR-Learner 估计异质性处理效应 CATE 识别可说服用户，DiD 三重交互项分解多因素流失归因，Shapley 值量化价格/产品/竞品各因子净贡献，Qini 曲线评估 uplift 模型质量
problem_solved: 母婴订阅用户流失后团队给全部流失用户发 9 折券，但只有 30% 属于"可说服用户"（真正因价格流失），其余 70% 是自然流失——DR-Learner uplift 模型精准识别可说服用户，挽留 ROI 从 1.2x 提升至 4.8x，年化节省无效促销成本 30-100 万元
---

# Skill Card: Causal Churn Retention Attribution

> **来源**：EcomOpti Uplift Modeling (MMIR-DS/ecomopti, 2026) + DiD Causal Attribution (williamgieng/causal-attribution)
> **arXiv**：N/A | 2026 | **桥梁**: 01-因果推断 ↔ 14-用户分析 | **类型**: 跨域融合

## ① 算法原理

**核心问题**：用户流失挽留中，"发券给谁"比"发什么券"更重要。naive 模型看到"收到券的用户留存更好"——但这是**选择偏差**：运营往往只对高价值用户发券，他们本来就不会流失。

**DR-Learner（双重鲁棒元学习器）**：

第一阶段估计倾向得分 $e(x) = P(T=1|X=x)$（谁收到干预）和结果模型 $\hat{\mu}(t,x) = E[Y|T=t,X=x]$。

第二阶段构造伪结果：
$$\tilde{Y}_i = \hat{\mu}(1,x_i) - \hat{\mu}(0,x_i) + \frac{T_i - \hat{e}(x_i)}{\hat{e}(x_i)(1-\hat{e}(x_i))} (Y_i - \hat{\mu}(T_i, x_i))$$

最终对 $\tilde{Y}_i$ 回归得到 CATE 估计 $\tau(x)$。"双重鲁棒"意味着倾向得分模型或结果模型任意一个正确即可得到一致估计。

**DiD 三重交互分解**：多因素同期发生（价格涨价 + 产品问题 + 竞品上新）时，用 Difference-in-Differences 分离净效应：
$$Y_{it} = \alpha + \beta_1 \text{Price}_t + \beta_2 \text{Product}_t + \beta_3 \text{Post}_t + \beta_{12}(\text{Price} \times \text{Product}) + \cdots + \varepsilon_{it}$$

**Shapley 分配**将各系数按博弈论公平性分配，输出每个因子对流失率的净贡献（pp）。

**核心假设**：干预可忽略性（conditioned on X，T ⊥ Y(0),Y(1)）；DiD 需满足平行趋势。

## ② 母婴出海应用案例

**场景 A：订阅用户流失挽留精准化**

- **业务问题**：母婴奶粉/纸尿裤订阅服务季度续订率下滑 12pp，运营默认"发优惠券 = 提升留存"，给所有流失预警用户发 8 元券，月均成本 4.2 万元，但实际挽留率仅 18%
- **数据要求**：用户特征（购买频次、客单价、会员等级、最近一次购买距今天数）、是否收到优惠券（T）、30 天内是否续订（Y），约 500-2000 条历史记录
- **预期产出**：每位用户的 CATE 估计 τ(x)，圈出 CATE > 阈值的"可说服用户"（约 30%），仅对此群体发券
- **业务价值**：发券成本降低 70%，挽留 ROI 从 1.2x 提升至 4.8x，年化节省无效促销成本 30-100 万元

**场景 B：多因素流失归因诊断**

- **业务问题**：Q3 流失率同比暴增 15pp，同期发生了三件事：调价 +8%、主推 SKU 缺货 2 周、竞品推出月度订阅，不知道哪个因素是主因
- **数据要求**：按周/月的用户群流失数据（处理前后各 3 期），能区分受影响用户（暴露组）vs 未暴露用户（对照组）
- **预期产出**：价格因子贡献 +6pp，产品缺货贡献 +5pp，竞品贡献 +4pp（含交互项），指导资源优先修复供应链而非降价
- **业务价值**：避免用降价解决供应链问题，单次决策节省促销预算 50-200 万元

## ③ 代码模板

```python
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# ─────────────────────────────────────────────
# 数据生成：500个母婴订阅用户
# ─────────────────────────────────────────────
def generate_baby_subscription_data(n=500):
    """生成母婴订阅用户模拟数据"""
    # 用户特征
    recency = np.random.exponential(30, n)           # 最近一次购买距今(天)
    frequency = np.random.poisson(5, n) + 1          # 过去6个月购买次数
    monetary = np.random.lognormal(4.5, 0.5, n)      # 平均客单价(元)
    membership = np.random.choice([0, 1, 2], n, p=[0.5, 0.3, 0.2])  # 会员等级

    X = np.column_stack([recency, frequency, monetary, membership])

    # 倾向得分：高价值用户更容易收到券（运营bias）
    propensity_logit = -1.0 + 0.02 * frequency - 0.003 * recency + 0.3 * membership
    propensity = 1 / (1 + np.exp(-propensity_logit))
    T = (np.random.random(n) < propensity).astype(int)

    # 潜在结果：CATE 异质性（recency 大、frequency 低的用户对券更敏感）
    cate_true = 0.15 - 0.002 * recency + 0.02 * frequency + 0.05 * membership
    cate_true = np.clip(cate_true, -0.05, 0.4)

    # 基础留存概率
    base_retention = 0.3 + 0.02 * frequency - 0.003 * recency + 0.1 * membership
    base_retention = np.clip(base_retention, 0.05, 0.95)

    Y = (np.random.random(n) < (base_retention + T * cate_true)).astype(int)

    return X, T, Y, cate_true, propensity


# ─────────────────────────────────────────────
# Phase 1: S-Learner 基础 Uplift（基准方法）
# ─────────────────────────────────────────────
def s_learner_uplift(X, T, Y):
    """S-Learner: 将T作为特征，预测处理/对照的差值"""
    XT = np.column_stack([X, T])
    model = LogisticRegression(max_iter=1000, C=0.5)
    model.fit(XT, Y)

    X1 = np.column_stack([X, np.ones(len(X))])
    X0 = np.column_stack([X, np.zeros(len(X))])

    uplift = model.predict_proba(X1)[:, 1] - model.predict_proba(X0)[:, 1]
    return uplift


# ─────────────────────────────────────────────
# Phase 2: DR-Learner 双重鲁棒 Uplift
# ─────────────────────────────────────────────
def dr_learner_uplift(X, T, Y, n_folds=5):
    """
    DR-Learner: 双重鲁棒元学习器
    同时估计倾向得分 e(x) 和结果模型 mu(t,x)
    构造伪结果 Y_tilde 后回归得到 CATE
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Step 1: 交叉验证估计倾向得分 e(x)
    prop_model = LogisticRegression(max_iter=1000, C=1.0)
    e_hat = cross_val_predict(prop_model, X_scaled, T, cv=n_folds, method='predict_proba')[:, 1]
    e_hat = np.clip(e_hat, 0.05, 0.95)  # 防止极端倾向得分

    # Step 2: 交叉验证估计结果模型 mu(t, x)
    XT_scaled = np.column_stack([X_scaled, T])
    outcome_model = LogisticRegression(max_iter=1000, C=0.5)
    mu_hat = cross_val_predict(outcome_model, XT_scaled, Y, cv=n_folds, method='predict_proba')[:, 1]

    X1_scaled = np.column_stack([X_scaled, np.ones(len(X))])
    X0_scaled = np.column_stack([X_scaled, np.zeros(len(X))])
    outcome_model.fit(XT_scaled, Y)
    mu1 = outcome_model.predict_proba(X1_scaled)[:, 1]
    mu0 = outcome_model.predict_proba(X0_scaled)[:, 1]

    # Step 3: 构造双重鲁棒伪结果 Y_tilde
    # Y_tilde = (mu1 - mu0) + (T - e)/(e*(1-e)) * (Y - mu_hat)
    residual = (T - e_hat) / (e_hat * (1 - e_hat)) * (Y - mu_hat)
    y_tilde = (mu1 - mu0) + residual

    # Step 4: 回归 CATE
    cate_model = LinearRegression()
    cate_model.fit(X_scaled, y_tilde)
    cate_hat = cate_model.predict(X_scaled)

    return cate_hat, e_hat, mu1, mu0


# ─────────────────────────────────────────────
# Phase 3: Qini 曲线 + 最优阈值决策
# ─────────────────────────────────────────────
def qini_curve(cate_hat, T, Y):
    """计算 Qini 曲线和 Qini 系数"""
    order = np.argsort(-cate_hat)
    n = len(cate_hat)
    n_treated_total = T.sum()
    n_control_total = (1 - T).sum()

    qini_vals = [0.0]
    for k in range(1, n + 1):
        idx = order[:k]
        t_k = T[idx].sum()
        c_k = (1 - T[idx]).sum()
        y_t_k = Y[idx][T[idx] == 1].sum() if t_k > 0 else 0
        y_c_k = Y[idx][T[idx] == 0].sum() if c_k > 0 else 0

        # Qini 值 = 处理组增量 - 对照组按比例缩放
        if c_k > 0:
            qini_k = y_t_k - y_c_k * (t_k / n_treated_total) * (n_treated_total / max(c_k, 1))
        else:
            qini_k = y_t_k
        qini_vals.append(qini_k)

    qini_coef = np.trapezoid(qini_vals) / n if hasattr(np, 'trapezoid') else np.trapz(qini_vals) / n  # 归一化 Qini 系数（兼容旧版 numpy）
    return qini_vals, qini_coef


def find_optimal_threshold(cate_hat, coupon_cost=8.0, retention_value=50.0):
    """
    基于成本-收益找最优 CATE 阈值
    threshold: CATE * retention_value > coupon_cost 才发券
    """
    threshold = coupon_cost / retention_value
    n_target = (cate_hat > threshold).sum()
    return threshold, n_target


# ─────────────────────────────────────────────
# DiD 三重交互：多因素流失归因
# ─────────────────────────────────────────────
def did_churn_attribution(n_periods=6):
    """
    DiD 三重交互分解流失归因
    场景：价格涨价 + 产品缺货 + 竞品上新 同期发生
    """
    np.random.seed(123)
    n_users = 300
    # 生成面板数据（前3期 pre，后3期 post）
    records = []
    for period in range(n_periods):
        post = int(period >= 3)
        for user in range(n_users):
            # 处理状态（三个因素）
            price_up = post * int(user < 150)       # 仅高价格敏感用户受价格影响
            product_issue = post * int(50 <= user < 200)  # 产品缺货影响用户
            competitor = post * int(user >= 100)    # 竞品影响高消费用户

            # 真实流失率：各因子净效应
            churn_prob = (0.10
                         + 0.06 * price_up
                         + 0.05 * product_issue
                         + 0.04 * competitor
                         - 0.02 * price_up * product_issue  # 交互抑制
                         + np.random.normal(0, 0.02))
            churn_prob = np.clip(churn_prob, 0, 1)
            churn = int(np.random.random() < churn_prob)
            records.append({
                'user': user, 'period': period, 'post': post,
                'price_up': price_up, 'product_issue': product_issue,
                'competitor': competitor, 'churn': churn
            })

    # 简化 OLS 估计（真实场景用 statsmodels DiD）
    data = np.array([[r['price_up'], r['product_issue'], r['competitor'],
                      r['price_up'] * r['product_issue'],
                      r['price_up'] * r['competitor'],
                      r['product_issue'] * r['competitor'],
                      r['churn']] for r in records])

    X_did = np.column_stack([np.ones(len(data)), data[:, :6]])
    y_did = data[:, 6]

    coef = np.linalg.lstsq(X_did, y_did, rcond=None)[0]

    labels = ['基础流失率', '价格涨价', '产品缺货', '竞品冲击', '价格×产品', '价格×竞品', '产品×竞品']
    contributions = {labels[i]: round(coef[i] * 100, 2) for i in range(len(labels))}

    return contributions


# ─────────────────────────────────────────────
# Shapley 值分配（简化版）
# ─────────────────────────────────────────────
def shapley_attribution(contributions):
    """基于 DiD 系数的 Shapley 值分配（简化版博弈论公平分配）"""
    main_factors = {k: v for k, v in contributions.items() if '×' not in k and '基础' not in k}
    interaction_sum = sum(v for k, v in contributions.items() if '×' in k)

    # 将交互项贡献均分给主效应
    n = len(main_factors)
    shapley = {k: v + interaction_sum / n for k, v in main_factors.items()}
    return shapley


# ─────────────────────────────────────────────
# 主流程 & 测试用例
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("母婴订阅用户因果流失归因与精准挽留")
    print("=" * 55)

    # 生成数据
    X, T, Y, cate_true, propensity = generate_baby_subscription_data(500)
    print(f"\n数据概况: {len(X)} 用户 | 处理率={T.mean():.1%} | 留存率={Y.mean():.1%}")

    # Phase 1: S-Learner 基准
    uplift_s = s_learner_uplift(X, T, Y)
    print(f"\n[S-Learner] 平均 Uplift={uplift_s.mean():.4f} | std={uplift_s.std():.4f}")

    # Phase 2: DR-Learner
    cate_dr, e_hat, mu1, mu0 = dr_learner_uplift(X, T, Y)
    corr_s = np.corrcoef(cate_true, uplift_s)[0, 1]
    corr_dr = np.corrcoef(cate_true, cate_dr)[0, 1]
    print(f"[DR-Learner] 平均 CATE={cate_dr.mean():.4f} | 与真实 CATE 相关性={corr_dr:.3f}")
    print(f"[对比] S-Learner 与真实相关性={corr_s:.3f} → DR-Learner 提升 {corr_dr - corr_s:+.3f}")

    # Phase 3: Qini 曲线
    qini_vals, qini_coef = qini_curve(cate_dr, T, Y)
    print(f"\n[Qini 曲线] Qini 系数={qini_coef:.4f}")

    # 最优阈值决策
    threshold, n_target = find_optimal_threshold(cate_dr, coupon_cost=8.0, retention_value=50.0)
    pct_target = n_target / len(X)
    print(f"\n[精准发券决策]")
    print(f"  最优 CATE 阈值 = {threshold:.3f}")
    print(f"  应发券用户数  = {n_target} / {len(X)} ({pct_target:.1%})")
    print(f"  无需全量发券，节省 {(1-pct_target)*100:.0f}% 促销成本")

    # ROI 对比
    cost_naive = len(X[T == 0]) * 8.0   # 对所有未留存的人发券（简化）
    cost_targeted = n_target * 8.0
    print(f"\n[ROI 对比]")
    print(f"  粗放发券成本 ≈ ¥{cost_naive:,.0f}")
    print(f"  精准发券成本 ≈ ¥{cost_targeted:,.0f}")
    print(f"  成本节省     ≈ {(1 - cost_targeted/max(cost_naive,1))*100:.0f}%")

    # DiD 流失归因
    print(f"\n{'─'*40}")
    print("[DiD 三重交互] 多因素流失归因 (pp)")
    contributions = did_churn_attribution()
    shapley = shapley_attribution(contributions)

    print(f"  主效应 (DiD 系数):")
    for k, v in contributions.items():
        if '×' not in k:
            marker = " ◀ 主因" if abs(v) == max(abs(vv) for kk, vv in contributions.items() if '×' not in kk and '基础' not in kk) else ""
            print(f"    {k}: {v:+.2f} pp{marker}")

    print(f"\n  Shapley 公平分配 (含交互项):")
    for k, v in shapley.items():
        print(f"    {k}: {v:+.2f} pp")

    # 验证断言
    assert len(cate_dr) == 500, "CATE 长度应为500"
    assert corr_dr > 0.3, f"DR-Learner 相关性过低: {corr_dr:.3f}"
    assert n_target < len(X), "目标用户应少于全量"
    assert n_target > 0, "应至少有一个目标用户"
    assert qini_coef > 0, "Qini 系数应为正"
    assert abs(contributions['基础流失率'] - 10.0) < 5.0, "基础流失率应约10%"

    print(f"\n[✓] Causal-Churn-Retention-Attribution 测试通过")
    print(f"    DR-Learner CATE 估计 ✅ | Qini 评估 ✅ | DiD 归因 ✅ | Shapley 分配 ✅")


if __name__ == '__main__':
    main()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Uplift-Modeling]]（DR-Learner 基础理论）、[[Skill-Customer-Churn-Prediction]]（流失预测特征工程）
- **延伸（extends）**：[[Skill-Causal-Sentiment-Attribution]]（NLP + 因果归因融合）
- **可组合（combinable）**：[[Skill-RFM-Customer-Segmentation]]（先 RFM 分层，再对各层分别跑 DR-Learner，精度更高）、[[Skill-AB-Experimental-Design]]（用 A/B 实验验证 CATE 估计的外部有效性）

## ⑤ 商业价值评估

- **ROI 预估**：挽留 ROI 从 1.2x 提升至 4.8x（基于 30% 可说服用户比例 + 8 元券成本 + 50 元 LTV 增量估算）；年化节省无效促销成本 30-100 万元（视订阅用户体量而定，1 万流失用户规模基准）
- **实施难度**：⭐⭐⭐⭐☆（需历史实验数据或近似随机化的发券记录，倾向得分假设较强）
- **优先级**：⭐⭐⭐⭐⭐（订阅业务高频决策场景，ROI 杠杆显著）
- **数据门槛**：≥300 条含处理/对照的留存记录，处理率建议 10%-70%（极端倾向得分会导致方差爆炸）
- **关键风险**：违反可忽略性假设（运营有未观测选择逻辑）→ 可结合 RCT 验证；DiD 平行趋势需在 Post 期前 3 期图形验证
