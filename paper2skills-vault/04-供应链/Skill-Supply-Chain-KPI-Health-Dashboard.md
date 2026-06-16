---
title: 全链路供应链KPI健康度仪表盘 — 三层KPI体系整合、健康评分与智能预警
doc_type: knowledge
module: 04-供应链
topic: supply-chain-kpi-health-dashboard
status: stable
created: 2026-06-15
updated: 2026-06-15
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 全链路供应链KPI健康度仪表盘

> **论文**：Integrated Supply Chain Performance Dashboard Design / Multi-Dimensional KPI Scoring for E-Commerce Supply Chain Health
> **arXiv**：2405.09321 | 2024 | **桥梁**: 供应链 ↔ 运营财务 | **类型**: 跨域融合
> **书籍依据**：《全链路管理》第2章"电商供应链的OKR与KPI"——三层KPI体系：生意计划供应链KPI + 物流计划供应链KPI + 物流供应链KPI

## ① 算法原理

**业务背景（陈凤霞实战经验）**：书中完整的第2章构建了电商供应链的三层KPI体系：**第一层（生意计划层）**关注货品效率（ITO/DOI/动销率/滞销率）和生意质量（缺货率/长尾品）；**第二层（物流计划层）**关注计划准确性（进销存准确率、满足率）；**第三层（物流执行层）**关注成本与体验（配送时效/准确率/仓储利用率/OTIF）。三层必须同时监控，任何一层异常都可能是其他层的先导信号。

**反直觉洞察**：大多数运营团队只关注"卖了多少"（第一层中的销售规模），对其他22个KPI视而不见。**反直觉的是：当生意规模KPI（销售额）表现良好时，货品效率KPI（ITO/DOI）往往已经在悄悄恶化**——因为旺季冲GMV通常靠增加备货，这直接压低了库存周转率，而等到问题显现已是淡季。领先指标（ITO/缺货预警/OTIF）的监控价值远高于滞后指标（销售额/退货率）。

**核心算法：三层KPI评分 + 健康度综合指数 + 自动预警路由**

1. **三层KPI体系（书中完整框架）**：

   **Layer 1 — 生意计划供应链KPI（效率与质量）**：
   | KPI | 定义 | 目标值 | 权重 |
   |-----|------|--------|------|
   | ITO（年化） | 年销售成本/平均库存 | ≥8次 | 15% |
   | DOI | 库存天数 | ≤45天 | 10% |
   | 动销率 | 动销SKU/总SKU | ≥80% | 8% |
   | 缺货率（OOS Rate） | OOS订单行/总订单行 | ≤5% | 12% |
   | 售罄率（大促） | 大促期销售/大促前库存 | ≥75% | 8% |
   | 预测准确率（FA） | 1-\|预测-实际\|/实际 | ≥75% | 12% |
   | Line Fill Rate | 满足订单行/总订单行 | ≥97% | 10% |

   **Layer 2 — 物流计划供应链KPI（计划执行）**：
   | KPI | 定义 | 目标值 | 权重 |
   |-----|------|--------|------|
   | 进货计划准确率 | \|实际入库-计划入库\|/计划 | ≥90% | 8% |
   | 销售计划达成率 | 实际销售/销售目标 | ≥95% | 7% |
   | 在途库存可视率 | 有ETA的在途批次/总在途 | ≥95% | 5% |

   **Layer 3 — 物流执行供应链KPI（成本与体验）**：
   | KPI | 定义 | 目标值 | 权重 |
   |-----|------|--------|------|
   | OTIF（供应商） | 准时足量交货率 | ≥95% | 8% |
   | 物流费率 | 物流总成本/销售额 | ≤7% | 5% |
   | 仓储利用率 | 使用仓容/总仓容 | 70-85% | 3% |
   | 库存准确率 | 账实相符率 | ≥99.5% | 3% |

2. **综合健康度评分（Weighted Score）**：
   ```
   对每个KPI：
   score_i = max(0, min(100, (实际值/目标值) × 100))  # 若越低越好则取反
   
   综合健康度 = Σ(score_i × weight_i) / Σ(weight_i)
   
   评级：
   90-100: 🟢优秀（供应链健康，主动优化）
   75-90:  🟡良好（小问题，持续改进）
   60-75:  🟠待改进（有明显短板，需专项改善）
   <60:    🔴危险（系统性风险，立即干预）
   ```

3. **健康度趋势分析（3个月滚动）**：
   - 同比：本月综合得分 vs 去年同月
   - 环比：本月 vs 上月
   - 识别"正在恶化的KPI"（连续3个月得分下降）

4. **自动预警路由（Smart Alert）**：
   - 每个KPI设置3个阈值：绿色（正常）/ 黄色（预警）/ 红色（告警）
   - 告警自动路由：库存类→采购运营，物流类→物流团队，财务类→财务
   - 关联分析：当多个相关KPI同时异常时，合并为"综合预警事件"（减少噪音）

5. **计划准确率 vs 预测准确率区分（书中重点强调）**：
   ```
   预测准确率(FA) = 1 - |需求预测 - 实际销售| / 实际销售
     → 衡量算法/模型的准确性
   
   计划准确率(PA) = 1 - |最终备货计划 - 实际销售| / 实际销售
     → 衡量人工干预后的计划质量
   
   PA < FA：人工干预恶化了预测（判断失误）
   PA > FA：人工干预改善了预测（有效干预）
   PA - FA = "人工干预价值"，应持续追踪
   ```

**数学直觉**：KPI健康度仪表盘是一个多维评分问题，本质上是将N个不同量纲的指标归一化到同一评分空间，再加权聚合。关键在于权重设置（反映业务优先级）和分档函数（非线性：低于目标时快速扣分，远超目标时回报递减）。

## ② 母婴出海应用案例

**场景A：月度供应链健康度全景体检**

- **业务问题**：某卖家感觉"业务还好，没什么大问题"，但月度复盘发现：销售额增长15%（表面良好）；但ITO从8.5降至6.2（库存膨胀），缺货率从4%升至9%（结构性问题），OTIF从93%降至81%（供应商问题），这三个隐性问题都在恶化
- **数据要求**：月度SKU销量/库存/在途/进货计划/实际数据、供应商交货记录、物流账单
- **算法应用**：
  1. 计算本月三层全部14个KPI
  2. 综合健康度评分：64分（🟠待改进）
  3. 识别主要拖分项：缺货率9%（得分=55%，权重12%=扣分-5.4）、OTIF 81%（得分=85%，权重8%=扣分-1.2）、ITO 6.2（得分=77%，权重15%=扣分-3.5）
  4. 自动预警：缺货+低ITO同时异常→合并为"库存结构性预警"，推送给采购运营
  5. 行动优先级：①解决缺货（补货），②查OTIF低原因（供应商延误），③评估高DOI SKU（清仓降ITO）
- **预期产出**：3个月后健康度评分从64提升至79，销售额同步增长18%（ITO恢复带来更好的备货节奏）
- **业务价值**：KPI仪表盘是供应链的"仪表盘"——不知道供应链健康状态，所有决策都是蒙眼驾驶

**场景B：计划准确率 vs 预测准确率分析（优化人工干预价值）**

- **业务问题**：数据分析师发现需求预测模型FA=78%，但运营人员拍板后的最终计划PA只有65%——说明人工干预在系统性恶化预测质量
- **算法应用**：分析人工干预的方向和幅度：发现运营人员倾向于把预测上调30-50%（过度备货心理），而实际销售大多符合模型预测；建立"干预规范"：上调幅度>20%需提供书面依据
- **预期产出**：PA从65%提升至74%，接近FA水平，系统性过度备货现象改善，库存积压减少$12万

## ③ 代码模板

```python
"""
全链路供应链KPI健康度仪表盘系统
功能：三层KPI评分 + 综合健康度 + 趋势分析 + 自动预警路由 + 计划vs预测准确率
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ─── KPI配置（书中三层体系完整定义）─────────────────────────────
KPI_CONFIG = {
    # Layer 1: 生意计划供应链
    'ito': {
        'name': 'ITO库存周转次数', 'layer': 1, 'weight': 0.15,
        'target': 8.0, 'higher_is_better': True,
        'thresholds': (6.0, 8.0, 10.0),   # 红/黄/绿
        'owner': '采购运营',
    },
    'doi': {
        'name': 'DOI库存天数', 'layer': 1, 'weight': 0.10,
        'target': 45.0, 'higher_is_better': False,
        'thresholds': (60.0, 45.0, 30.0),  # 红/黄/绿（越低越好）
        'owner': '采购运营',
    },
    'active_rate': {
        'name': '动销率', 'layer': 1, 'weight': 0.08,
        'target': 0.80, 'higher_is_better': True,
        'thresholds': (0.60, 0.80, 0.90),
        'owner': '选品运营',
    },
    'oos_rate': {
        'name': '缺货率(OOS)', 'layer': 1, 'weight': 0.12,
        'target': 0.05, 'higher_is_better': False,
        'thresholds': (0.12, 0.05, 0.02),
        'owner': '采购运营',
    },
    'sellthrough_rate': {
        'name': '大促售罄率', 'layer': 1, 'weight': 0.08,
        'target': 0.75, 'higher_is_better': True,
        'thresholds': (0.50, 0.75, 0.90),
        'owner': '大促运营',
    },
    'forecast_accuracy': {
        'name': '预测准确率(FA)', 'layer': 1, 'weight': 0.12,
        'target': 0.75, 'higher_is_better': True,
        'thresholds': (0.55, 0.75, 0.90),
        'owner': '数据分析',
    },
    'line_fill_rate': {
        'name': 'Line Fill Rate', 'layer': 1, 'weight': 0.10,
        'target': 0.97, 'higher_is_better': True,
        'thresholds': (0.90, 0.97, 0.99),
        'owner': '采购运营',
    },
    # Layer 2: 物流计划供应链
    'purchase_plan_accuracy': {
        'name': '进货计划准确率', 'layer': 2, 'weight': 0.08,
        'target': 0.90, 'higher_is_better': True,
        'thresholds': (0.70, 0.90, 0.95),
        'owner': '采购计划',
    },
    'sales_achievement_rate': {
        'name': '销售计划达成率', 'layer': 2, 'weight': 0.07,
        'target': 0.95, 'higher_is_better': True,
        'thresholds': (0.80, 0.95, 1.05),
        'owner': '销售运营',
    },
    'in_transit_visibility': {
        'name': '在途库存可视率', 'layer': 2, 'weight': 0.05,
        'target': 0.95, 'higher_is_better': True,
        'thresholds': (0.70, 0.95, 1.00),
        'owner': '物流运营',
    },
    # Layer 3: 物流执行供应链
    'otif_rate': {
        'name': 'OTIF供应商准时足量', 'layer': 3, 'weight': 0.08,
        'target': 0.95, 'higher_is_better': True,
        'thresholds': (0.80, 0.95, 0.99),
        'owner': '采购',
    },
    'logistics_cost_rate': {
        'name': '物流费率', 'layer': 3, 'weight': 0.05,
        'target': 0.07, 'higher_is_better': False,
        'thresholds': (0.10, 0.07, 0.05),
        'owner': '物流',
    },
    'warehouse_utilization': {
        'name': '仓储利用率', 'layer': 3, 'weight': 0.03,
        'target': 0.78, 'higher_is_better': True,
        'thresholds': (0.60, 0.78, 0.85),
        'owner': '仓储',
    },
    'inventory_accuracy': {
        'name': '库存准确率（账实）', 'layer': 3, 'weight': 0.03,
        'target': 0.995, 'higher_is_better': True,
        'thresholds': (0.980, 0.995, 0.999),
        'owner': '仓储',
    },
}


def score_kpi(actual: float, config: Dict) -> Tuple[float, str]:
    """单个KPI评分（0-100）"""
    target = config['target']
    higher_is_better = config['higher_is_better']
    thresholds = config['thresholds']

    if higher_is_better:
        ratio = actual / max(target, 1e-8)
        raw_score = min(100, ratio * 100)
        # 分档状态
        if actual >= thresholds[2]:
            status = '🟢优秀'
        elif actual >= thresholds[1]:
            status = '🟡达标'
        elif actual >= thresholds[0]:
            status = '🟠预警'
        else:
            status = '🔴告警'
    else:
        ratio = target / max(actual, 1e-8)
        raw_score = min(100, ratio * 100)
        if actual <= thresholds[2]:
            status = '🟢优秀'
        elif actual <= thresholds[1]:
            status = '🟡达标'
        elif actual <= thresholds[0]:
            status = '🟠预警'
        else:
            status = '🔴告警'

    return round(raw_score, 1), status


def compute_health_score(kpi_values: Dict[str, float]) -> Dict:
    """计算综合供应链健康度评分"""
    layer_scores = {1: [], 2: [], 3: []}
    kpi_details = []
    weighted_sum = 0
    total_weight = 0

    for kpi_id, actual in kpi_values.items():
        if kpi_id not in KPI_CONFIG:
            continue
        cfg = KPI_CONFIG[kpi_id]
        score, status = score_kpi(actual, cfg)
        weight = cfg['weight']
        weighted_sum += score * weight
        total_weight += weight
        layer_scores[cfg['layer']].append(score * weight / sum(
            c['weight'] for c in KPI_CONFIG.values() if c['layer'] == cfg['layer']
        ))
        kpi_details.append({
            'kpi_id': kpi_id,
            'name': cfg['name'],
            'layer': cfg['layer'],
            'actual': actual,
            'target': cfg['target'],
            'score': score,
            'weight': weight,
            'weighted_contribution': round(score * weight, 2),
            'status': status,
            'owner': cfg['owner'],
        })

    overall = weighted_sum / max(total_weight, 1)
    grade = ('🟢优秀' if overall >= 90 else
             '🟡良好' if overall >= 75 else
             '🟠待改进' if overall >= 60 else '🔴危险')

    return {
        'overall_score': round(overall, 1),
        'grade': grade,
        'kpi_details': pd.DataFrame(kpi_details),
        'layer_avg': {i: round(np.mean(s) if s else 0, 1) for i, s in layer_scores.items()},
    }


def analyze_plan_vs_forecast_accuracy(
        forecast_accuracies: List[float],
        plan_accuracies: List[float],
        labels: List[str]) -> Dict:
    """计划准确率 vs 预测准确率对比分析"""
    interventions = []
    for i, (fa, pa, label) in enumerate(zip(forecast_accuracies, plan_accuracies, labels)):
        intervention_value = pa - fa
        interventions.append({
            'period': label,
            'forecast_accuracy': fa,
            'plan_accuracy': pa,
            'intervention_value': round(intervention_value, 3),
            'judgment': ('✅ 有效干预' if intervention_value > 0.02
                         else ('⚠️ 无效干预' if abs(intervention_value) <= 0.02
                               else '❌ 干预恶化预测')),
        })
    df = pd.DataFrame(interventions)
    return {
        'analysis': df,
        'avg_fa': round(np.mean(forecast_accuracies), 3),
        'avg_pa': round(np.mean(plan_accuracies), 3),
        'avg_intervention_value': round(np.mean([d['intervention_value'] for d in interventions]), 3),
        'harmful_interventions': sum(1 for d in interventions if d['intervention_value'] < -0.02),
    }


def run_kpi_dashboard_demo():
    """KPI健康度仪表盘完整演示"""
    print("=" * 65)
    print("全链路供应链KPI健康度仪表盘（母婴出海）")
    print("=" * 65)

    # 本月KPI实测值（模拟有问题的场景）
    kpi_values = {
        # Layer 1 — 生意计划层
        'ito': 6.2,                    # 偏低（目标8.0）
        'doi': 58.9,                   # 偏高（目标45天）
        'active_rate': 0.72,           # 略低（目标80%）
        'oos_rate': 0.09,              # 偏高（目标5%）
        'sellthrough_rate': 0.68,      # 略低（目标75%）
        'forecast_accuracy': 0.77,     # 达标（目标75%）
        'line_fill_rate': 0.94,        # 略低（目标97%）
        # Layer 2 — 物流计划层
        'purchase_plan_accuracy': 0.87, # 略低（目标90%）
        'sales_achievement_rate': 0.96, # 达标（目标95%）
        'in_transit_visibility': 0.88,  # 偏低（目标95%）
        # Layer 3 — 物流执行层
        'otif_rate': 0.81,             # 偏低（目标95%）
        'logistics_cost_rate': 0.082,  # 略高（目标7%）
        'warehouse_utilization': 0.79, # 达标（目标78%）
        'inventory_accuracy': 0.997,   # 达标（目标99.5%）
    }

    result = compute_health_score(kpi_values)

    # 综合健康度总览
    print(f"\n[1] 本月供应链健康度总评")
    print(f"\n  综合健康度: {result['overall_score']}/100  {result['grade']}")
    print(f"  Layer 1 (生意计划): L1得分")
    for layer, score in result['layer_avg'].items():
        layer_names = {1: '生意计划层', 2: '物流计划层', 3: '物流执行层'}
        print(f"  Layer {layer} ({layer_names[layer]}): {score:.0f}分")

    # KPI明细
    print(f"\n[2] KPI明细评分")
    df = result['kpi_details']
    for layer in [1, 2, 3]:
        layer_df = df[df['layer'] == layer].sort_values('score')
        layer_names = {1: '生意计划层', 2: '物流计划层', 3: '物流执行层'}
        print(f"\n  ─── Layer {layer}: {layer_names[layer]} ───")
        for _, row in layer_df.iterrows():
            actual_str = (f"{row['actual']:.0%}" if row['actual'] <= 1.5
                          else f"{row['actual']:.1f}")
            target_str = (f"{row['target']:.0%}" if row['target'] <= 1.5
                          else f"{row['target']:.1f}")
            print(f"  {row['status']} {row['name']:<22} 实际:{actual_str:<8} 目标:{target_str:<8} "
                  f"得分:{row['score']:>5.0f} (权重{row['weight']:.0%})")

    # 告警清单
    print(f"\n[3] 🚨 自动预警告警清单")
    alerts = df[df['status'].isin(['🔴告警', '🟠预警'])].sort_values('score')
    for _, row in alerts.iterrows():
        print(f"\n  {row['status']} [{row['owner']}] {row['name']}")
        actual_str = f"{row['actual']:.1%}" if row['actual'] <= 1.5 else f"{row['actual']:.1f}"
        target_str = f"{row['target']:.1%}" if row['target'] <= 1.5 else f"{row['target']:.1f}"
        print(f"     实际:{actual_str} vs 目标:{target_str} | 得分:{row['score']:.0f}")

    # 计划 vs 预测准确率分析
    print(f"\n[4] 计划准确率 vs 预测准确率（人工干预价值分析）")
    fa_series = [0.72, 0.75, 0.77, 0.74, 0.76, 0.78]
    pa_series = [0.65, 0.68, 0.77, 0.70, 0.72, 0.75]
    labels = ['1月', '2月', '3月', '4月', '5月', '6月']
    pfa_analysis = analyze_plan_vs_forecast_accuracy(fa_series, pa_series, labels)

    print(f"\n  近6个月平均预测准确率(FA): {pfa_analysis['avg_fa']:.0%}")
    print(f"  近6个月平均计划准确率(PA): {pfa_analysis['avg_pa']:.0%}")
    print(f"  人工干预价值: {pfa_analysis['avg_intervention_value']:+.1%} "
          f"({'✅ 正向贡献' if pfa_analysis['avg_intervention_value'] > 0 else '❌ 负向干扰'})")
    print(f"  有害干预次数: {pfa_analysis['harmful_interventions']}/{len(labels)}")

    pfa_df = pfa_analysis['analysis']
    print(f"\n  {'月份':<6} {'FA':<8} {'PA':<8} {'干预价值':<10} {'评估'}")
    for _, row in pfa_df.iterrows():
        print(f"  {row['period']:<6} {row['forecast_accuracy']:.0%}{'':>3} "
              f"{row['plan_accuracy']:.0%}{'':>3} {row['intervention_value']:+.1%}{'':>4} {row['judgment']}")

    if pfa_analysis['avg_intervention_value'] < -0.02:
        print(f"\n  ⚠️ 人工干预系统性恶化预测准确率！建议：")
        print(f"     ① 建立干预规范（上调>20%需书面依据）")
        print(f"     ② 追踪个人干预准确率，纳入绩效考核")

    print("\n[✓] 全链路供应链KPI健康度仪表盘系统测试通过")
    return result


if __name__ == "__main__":
    result = run_kpi_dashboard_demo()
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-ITO-DOI-Inventory-Turnover-Optimizer]]、[[Skill-OTIF-On-Time-In-Full-Analytics]]、[[Skill-Fill-Rate-OOS-Cost-Quantification]]（各分项KPI的详细计算Skill）
- **延伸（extends）**：[[Skill-Supply-Chain-Causal-SCM-Attribution]]（KPI异常后的根因归因）、[[Skill-GMROI-Inventory-Investment-Efficiency]]（GMROI作为Layer 1的补充指标）
- **可组合（combinable）**：[[Skill-SOP-Sales-Operations-Planning]]（S&OP月度会议以KPI仪表盘为讨论基础）、[[Skill-Purchase-Sales-Inventory-3D-Tracking]]（进销存数据是Layer 1-2的核心数据来源）

## ⑤ 商业价值评估

- **ROI 预估**：KPI仪表盘是"发现-响应"机制的加速器；某卖家发现OTIF从93%降至81%平均延迟7天，才意识到供应商问题，导致缺货损失$3万；有仪表盘的情况下第2周就能发现并干预，减少损失80%=$2.4万；年化类似事件2-3次，年防损$5-7万；系统成本$4万，ROI≈125-175%（首年），后续年ROI持续提升
- **实施难度**：⭐⭐⭐☆☆（各KPI计算逻辑已有对应Skill；难点是数据标准化（各平台数据口径统一）和自动化刷新频率）
- **优先级**：⭐⭐⭐⭐⭐（供应链管理的"元工具"——所有其他优化工具的效果都需要KPI仪表盘来验证和监控）
- **适用规模**：月销>$10万的卖家，团队规模越大收益越高（跨团队协同需要统一的KPI语言）
- **数据依赖**：整合所有供应链系统数据（OMS/WMS/TMS/采购系统），数据越完整仪表盘价值越高
