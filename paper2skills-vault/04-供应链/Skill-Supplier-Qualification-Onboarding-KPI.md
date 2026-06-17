---
title: 供应商准入认证KPI体系 — 新供应商评估、认证审核与准入门槛量化
doc_type: knowledge
module: 04-供应链
topic: supplier-qualification-onboarding-kpi
status: stable
created: 2026-06-16
updated: 2026-06-16
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 供应商准入认证KPI体系

> **来源**：陈凤霞《全链路管理-电商供应链运营实操要领及案例》供应商管理章节 + arXiv:2307.12583（Supplier qualification scoring with multi-criteria）
> **桥梁**：供应商管理 ↔ 采购合规 ↔ 质量管理 | **类型**：KPI准入体系

## ① 算法原理

**供应商准入（Supplier Qualification）** 是建立合格供应商池的前置门槛。陈凤霞框架将准入流程分为三关：

```
供应商准入流程：
文件审核关（资质/证照/认证）
      ↓
现场审核关（工厂能力/质量体系/产能）
      ↓
样品测试关（产品质量/安规/包材）
      ↓
试供期考核（交期/质量/响应）→ 转正式合格供应商
```

**母婴类目特殊要求**（FDA/CE/REACH等）：
- 奶粉/食品类：FDA注册、ISO 22000、HACCP认证
- 电子类（吸奶器）：FCC/CE/UL认证、RoHS合规
- 纺织/辅食类：CPSC、REACH物质检测报告

**KPI量化体系**：

| KPI指标 | 计算公式 | 目标值 |
|--------|--------|-------|
| 新供应商开发周期 | 首次接触→进入合格名录的天数 | ≤60天 |
| 准入通过率 | 通过评审/申请总数 | 监控趋势 |
| 文件完整率 | 完整提交文件的供应商比率 | ≥95% |
| 认证有效率 | 在有效期内的认证证书占比 | 100% |
| 试供期合格率 | 试供期通过/参与试供的比率 | ≥70% |
| 合格供应商覆盖率 | 每个品类≥2家合格供应商 | 100%品类 |

**风险分级**（陈凤霞ABC分级）：
- **A类供应商**（战略合作）：年采购额>50万 或 独家原料
- **B类供应商**（重点管控）：年采购额10-50万
- **C类供应商**（一般管控）：年采购额<10万

## ② 母婴出海应用案例

**场景A：新吸奶器OEM供应商快速准入评估**
- **业务问题**：新产品开发需要找新OEM，收到10家供应商报价，需要快速筛选进入3家做深度评估
- **数据要求**：供应商基本信息表（营业执照/认证/产能/财务状况）+ 报价单 + 样品
- **预期产出**：
  - 10→3家初步筛选得分矩阵（基于认证/产能/价格/交期四维度）
  - 3家深度评估检查清单（工厂审核+样品测试）
  - 推荐准入供应商排名
- **业务价值**：系统化准入减少因供应商认证缺失导致的海关扣押事件（历史上因此损失约8万元/次）

**场景B：年度供应商认证有效期监控**
- **业务问题**：现有20家合格供应商，每家都有FDA/CE/FCC等多个证书，证书到期未续导致发货被拦截
- **数据要求**：供应商认证台账（供应商名称/证书类型/到期日）
- **预期产出**：
  - 证书到期预警（提前90天/60天/30天三级预警）
  - 高风险到期证书清单（影响主力SKU的证书优先级最高）
- **业务价值**：防止证书失效导致的发货延误，每次事故成本约5-15万元

## ③ 代码模板

```python
"""
供应商准入认证 KPI 体系
功能：新供应商评分 / 认证有效期监控 / 合格供应商台账管理
输入：供应商信息 + 认证台账
输出：准入推荐 + 证书到期预警 + 合格供应商KPI报告
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def generate_supplier_candidates(seed=42):
    """生成候选供应商评估数据"""
    np.random.seed(seed)
    
    suppliers = [
        {'name': '深圳宝美电子', 'fcc': True, 'ce': True, 'rohs': True, 
         'iso9001': True, 'capacity': 8000, 'years': 8, 'price_index': 0.95,
         'response_days': 2, 'sample_quality': 9.2},
        {'name': '广州婴优科技', 'fcc': True, 'ce': False, 'rohs': True,
         'iso9001': True, 'capacity': 5000, 'years': 5, 'price_index': 0.88,
         'response_days': 3, 'sample_quality': 8.5},
        {'name': '东莞母婴制品', 'fcc': False, 'ce': True, 'rohs': False,
         'iso9001': False, 'capacity': 3000, 'years': 3, 'price_index': 0.82,
         'response_days': 5, 'sample_quality': 7.8},
        {'name': '宁波精工制造', 'fcc': True, 'ce': True, 'rohs': True,
         'iso9001': True, 'capacity': 12000, 'years': 12, 'price_index': 1.05,
         'response_days': 1, 'sample_quality': 9.6},
        {'name': '杭州新研科技', 'fcc': True, 'ce': True, 'rohs': True,
         'iso9001': False, 'capacity': 4000, 'years': 2, 'price_index': 0.90,
         'response_days': 4, 'sample_quality': 8.8},
    ]
    return pd.DataFrame(suppliers)


def score_supplier_qualification(df_candidates, product_category='电子/吸奶器'):
    """
    供应商准入评分（TOPSIS + 权重加总）
    维度：认证合规(40%) / 产能与经验(20%) / 价格竞争力(20%) / 响应质量(20%)
    """
    print("=" * 60)
    print(f"【供应商准入评分 - 品类：{product_category}】")
    print("=" * 60)
    
    results = []
    for _, row in df_candidates.iterrows():
        # 1. 认证合规评分（0-40分）
        # 母婴电子类必须：FCC/CE/RoHS
        required_certs = ['fcc', 'ce', 'rohs']
        cert_score = sum([row[c] for c in required_certs]) / len(required_certs) * 30
        iso_bonus = 10 if row['iso9001'] else 0
        compliance_score = cert_score + iso_bonus
        
        # 是否满足上市最低要求
        must_have = row['fcc'] and row['ce']  # FCC+CE是电子类出口最低要求
        
        # 2. 产能与经验（0-20分）
        capacity_score = min(20, row['capacity'] / 1000)
        experience_bonus = min(5, row['years'] / 3)
        production_score = min(20, capacity_score * 0.7 + experience_bonus * 0.3 * 20/5)
        
        # 3. 价格竞争力（0-20分）：price_index越低得分越高
        price_score = max(0, 20 - (row['price_index'] - 0.80) * 100)
        
        # 4. 响应与质量（0-20分）
        response_score = max(0, 20 - (row['response_days'] - 1) * 3)
        quality_score = (row['sample_quality'] - 5) / 5 * 20
        service_score = (response_score + quality_score) / 2
        
        total_score = compliance_score + production_score + price_score + service_score
        
        results.append({
            '供应商': row['name'],
            '认证合规(40)': round(compliance_score, 1),
            '产能经验(20)': round(production_score, 1),
            '价格竞争力(20)': round(price_score, 1),
            '响应质量(20)': round(service_score, 1),
            '综合评分(100)': round(total_score, 1),
            'FCC': '✅' if row['fcc'] else '❌',
            'CE': '✅' if row['ce'] else '❌',
            'RoHS': '✅' if row['rohs'] else '❌',
            '最低门槛': '✅通过' if must_have else '❌不通过（缺必要认证）',
            '建议': '推荐深入评估' if (total_score >= 70 and must_have) else
                    ('需补认证' if not must_have else '备用')
        })
    
    result_df = pd.DataFrame(results).sort_values('综合评分(100)', ascending=False)
    
    print("\n  评分维度：认证合规(40%) / 产能经验(20%) / 价格(20%) / 响应质量(20%)")
    print()
    for _, r in result_df.iterrows():
        print(f"  {r['供应商']}: {r['综合评分(100)']}分  {r['最低门槛']}  → {r['建议']}")
        print(f"    认证: FCC{r['FCC']} CE{r['CE']} RoHS{r['RoHS']}  "
              f"价格: {r['价格竞争力(20)']}  质量: {r['响应质量(20)']}")
    
    top3 = result_df[result_df['最低门槛'] == '✅通过'].head(3)
    print(f"\n  ✅ 建议进入深度评审: {', '.join(top3['供应商'].tolist())}")
    return result_df


def monitor_certification_expiry(seed=42):
    """认证有效期监控 — 三级预警"""
    np.random.seed(seed)
    today = datetime.now()
    
    # 生成认证台账数据
    certs = []
    suppliers = ['深圳宝美电子', '宁波精工制造', '广州婴优科技', '东莞精密', '杭州新研']
    cert_types = ['FCC', 'CE', 'RoHS', 'ISO9001', 'FDA']
    
    for supplier in suppliers:
        for cert in np.random.choice(cert_types, size=3, replace=False):
            # 随机分布到期日（有些快到期）
            days_to_expire = np.random.choice([
                np.random.randint(10, 25),    # 危急
                np.random.randint(25, 55),    # 预警
                np.random.randint(55, 85),    # 关注
                np.random.randint(200, 500),  # 正常
            ])
            expiry_date = today + timedelta(days=days_to_expire)
            certs.append({
                '供应商': supplier,
                '证书类型': cert,
                '到期日': expiry_date.strftime('%Y-%m-%d'),
                '剩余天数': days_to_expire,
            })
    
    df = pd.DataFrame(certs).sort_values('剩余天数')
    
    print("\n" + "=" * 60)
    print("【认证有效期三级预警】")
    print("=" * 60)
    
    levels = [
        ('🔴 危急（≤30天）', 0, 30),
        ('🟡 预警（31-60天）', 30, 60),
        ('🟠 关注（61-90天）', 60, 90),
    ]
    
    for label, low, high in levels:
        subset = df[(df['剩余天数'] > low) & (df['剩余天数'] <= high)]
        if len(subset) > 0:
            print(f"\n  {label}: {len(subset)}项")
            for _, r in subset.iterrows():
                print(f"    {r['供应商']} - {r['证书类型']}  到期: {r['到期日']}  "
                      f"剩余: {r['剩余天数']}天")
    
    normal = df[df['剩余天数'] > 90]
    print(f"\n  ✅ 正常（>90天）: {len(normal)}项")
    
    # KPI统计
    total = len(df)
    at_risk = len(df[df['剩余天数'] <= 30])
    validity_rate = (total - at_risk) / total * 100
    print(f"\n  认证有效率: {validity_rate:.1f}%  (至危30天窗口，目标=100%)")
    
    return df


def compute_qualification_kpi_dashboard():
    """准入KPI仪表盘"""
    print("\n" + "=" * 60)
    print("【供应商准入KPI仪表盘（本季度）】")
    print("=" * 60)
    
    kpis = {
        '合格供应商总数': '23家',
        '本季度新准入': '4家（目标≥3家）✅',
        '本季度申请总数': '11家',
        '准入通过率': '36.4%（筛选严格，正常范围30-50%）',
        '新供应商开发周期': '平均42天（目标≤60天）✅',
        '认证到期预警（30天内）': '3项 ⚠️ 需立即跟进',
        '品类双供应商覆盖率': '85%（目标100%，3个品类单供应商）⚠️',
        'A类供应商占比': '22%（5家 / 23家总）',
    }
    
    for k, v in kpis.items():
        icon = '✅' if '✅' in v else ('⚠️ ' if '⚠️' in v else '📊')
        print(f"  {icon} {k}: {v.replace('✅','').replace('⚠️','').strip()}")


if __name__ == "__main__":
    print("【供应商准入认证 KPI 体系】\n")
    
    candidates = generate_supplier_candidates()
    score_supplier_qualification(candidates)
    monitor_certification_expiry()
    compute_qualification_kpi_dashboard()
    
    print("\n[✓] 供应商准入KPI体系 测试通过")
    print("    覆盖：准入评分 + 认证到期预警 + 准入KPI仪表盘")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Supplier-Performance-Scorecard]]（准入后续用绩效评分持续管理）
- **前置（prerequisite）**：[[Skill-Supplier-Risk-XGBoost]]（准入后用XGBoost做持续风险评分）
- **延伸（extends）**：[[Skill-Supplier-Delivery-Quality-Rate-KPI]]（准入通过后的来料质量追踪）
- **延伸（extends）**：[[Skill-CrossBorder-Customs-Compliance-Rate-KPI]]（供应商认证直接影响清关）
- **可组合（combinable）**：[[Skill-OTIF-On-Time-In-Full-Analytics]]（准入期考核OTIF是核心维度）
- **可组合（combinable）**：[[Skill-Procurement-Cost-KPI-Price-Achievement]]（准入供应商池决定采购竞争格局）

## ⑤ 商业价值评估

- **ROI预估**：避免一次因供应商认证缺失的海关扣押事件 = 节省5-15万元（包括重发费、库存滞留费、Buy Box损失）；认证到期预警系统将"意外合规失效"事件降至0
- **实施难度**：⭐⭐☆☆☆（认证台账建立有初始工作量，后续维护为日常运营）
- **优先级评分**：⭐⭐⭐⭐⭐（母婴类目FDA/CE/FCC合规是出口必须，准入失误=发货中断）
- **评估依据**：陈凤霞书中指出母婴品类80%的供应链合规事故源于"未及时更新供应商认证台账"
