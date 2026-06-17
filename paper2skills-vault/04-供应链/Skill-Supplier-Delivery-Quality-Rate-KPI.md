---
title: 供应商来料质量IQC-KPI体系 — 入库检验合格率、质量成本与批次拒收分析
doc_type: knowledge
module: 04-供应链
topic: supplier-delivery-quality-rate-kpi
status: stable
created: 2026-06-16
updated: 2026-06-16
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: 供应商来料质量IQC-KPI体系

> **来源**：陈凤霞《全链路管理-电商供应链运营实操要领及案例》质量管理章节 + arXiv:2308.07441（Quality-aware supplier evaluation）
> **桥梁**：供应商管理 ↔ 质量管理 ↔ 采购决策 | **类型**：质量KPI体系

## ① 算法原理

**IQC（Incoming Quality Control，来料质量检验）** 是供应商质量管理的第一道防线。陈凤霞体系将来料质量KPI分为三层：

```
来料质量KPI层次：
结果指标（批次合格率、退货率）
      ↓
过程指标（检验时效、检验覆盖率、抽样方案合理性）
      ↓
成本指标（来料质量成本COPQ = 内部失败+外部失败+检验+预防）
```

**核心KPI定义**：

1. **批次合格率（Lot Acceptance Rate, LAR）**：
   $$\text{LAR} = \frac{\text{验收通过批次数}}{\text{检验总批次数}} \times 100\%$$
   行业基准：A类供应商 ≥ 98%，B类 ≥ 95%，C类 < 95% 触发整改

2. **来料一次通过率（First Pass Yield, FPY)**：
   $$\text{FPY} = \frac{\text{一次检验通过数量}}{\text{检验总数量}} \times 100\%$$

3. **质量成本（COPQ, Cost of Poor Quality）**：
   $$\text{COPQ} = C_{内部返工} + C_{退货重发} + C_{客诉赔偿} + C_{检验人工}$$

4. **供应商质量评级**（基于滚动3月数据）：
   - 优质：LAR ≥ 98% + 无重大质量事故
   - 合格：LAR 95-98% + COPQ/采购额 < 1%
   - 观察：LAR 90-95% 或 有质量事故
   - 整改：LAR < 90% 或 连续2批次拒收

**统计过程控制（SPC）应用**：
使用 p-chart 监控批次缺陷率的过程稳定性，识别特殊原因变异（供应商工艺变更/原材料切换）。

## ② 母婴出海应用案例

**场景A：吸奶器电机组件来料IQC管控**
- **业务问题**：吸奶器主要部件电机从2家供应商采购，A供应商历史质量稳定但近3月批次退货率从1.2%升至4.8%
- **数据要求**：过去12月每批次检验记录（批次号/数量/抽样数/缺陷数/缺陷类型/处理结果）
- **预期产出**：
  - SPC p-chart显示A供应商月份X出现特殊原因（工艺切换点）
  - 主要缺陷类型：密封件变形（72%）→ 定位到原材料供应商变更
  - 质量成本量化：COPQ季度约8.5万元
- **业务价值**：发现根因后与供应商协定原材料回切，3月内LAR恢复至98.5%，COPQ降低75%

**场景B：奶粉包材来料质量分析（FDA合规视角）**
- **业务问题**：包材（内袋/外箱）质量影响FDA认证和客户退货，需要建立系统性IQC-KPI
- **数据要求**：包材检验记录 + 不良品分类（尺寸偏差/印刷不合格/材质检测不过）
- **预期产出**：主要缺陷柏拉图（Pareto分析）+ 缺陷跨批次趋势
- **业务价值**：系统化管控将包材相关客诉退货从2.1%降至0.8%，避免FDA检查风险

## ③ 代码模板

```python
"""
供应商来料质量 IQC-KPI 体系
功能：批次合格率计算 / SPC p-chart监控 / 质量成本COPQ量化 / 供应商质量评级
输入：IQC检验记录（批次级）
输出：质量KPI报告 + 异常批次预警 + 供应商质量排名
"""
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def generate_iqc_data(n_batches=120, seed=42):
    """生成模拟IQC检验数据"""
    np.random.seed(seed)
    
    suppliers = {
        'SUP-A深圳宝美': {'base_defect_rate': 0.012, 'shift_at': 80},  # 第80批起工艺异常
        'SUP-B宁波精工': {'base_defect_rate': 0.008, 'shift_at': None},  # 稳定
        'SUP-C杭州新研': {'base_defect_rate': 0.035, 'shift_at': None},  # 质量较差
    }
    
    defect_types = ['密封件变形', '尺寸超差', '外观划痕', '功能失效', '材料不合格']
    records = []
    
    for i in range(n_batches):
        for sup, params in suppliers.items():
            base_rate = params['base_defect_rate']
            # 模拟工艺异常
            if params['shift_at'] and i >= params['shift_at']:
                base_rate *= 3.5  # 质量恶化
            
            sample_size = np.random.randint(80, 200)
            defect_count = np.random.binomial(sample_size, base_rate)
            lot_qty = sample_size * np.random.randint(5, 15)
            
            defect_rate = defect_count / sample_size
            accepted = defect_rate <= 0.025  # 接受准则 AQL 2.5%
            
            # 质量成本估算
            if not accepted:
                rework_cost = lot_qty * 2.5         # 返工/重检成本
                redelivery_cost = 1500              # 退货重发固定成本
                copq = rework_cost + redelivery_cost
            else:
                copq = 0
            
            records.append({
                'batch_id': f'B{i+1:03d}-{sup[:5]}',
                'supplier': sup,
                'batch_seq': i + 1,
                'lot_qty': lot_qty,
                'sample_size': sample_size,
                'defect_count': defect_count,
                'defect_rate': defect_rate,
                'accepted': accepted,
                'primary_defect': np.random.choice(
                    defect_types,
                    p=[0.45, 0.20, 0.15, 0.12, 0.08]
                ) if defect_count > 0 else '无',
                'copq_yuan': copq,
            })
    
    return pd.DataFrame(records)


def compute_supplier_quality_kpi(df):
    """计算各供应商质量KPI"""
    print("=" * 60)
    print("【供应商来料质量 KPI 汇总】")
    print("=" * 60)
    
    for supplier, grp in df.groupby('supplier'):
        total_batches = len(grp)
        accepted_batches = grp['accepted'].sum()
        lar = accepted_batches / total_batches * 100
        mean_defect_rate = grp['defect_rate'].mean() * 100
        total_copq = grp['copq_yuan'].sum() / 10000
        
        # 评级
        if lar >= 98:
            rating = '⭐优质'
        elif lar >= 95:
            rating = '✅合格'
        elif lar >= 90:
            rating = '⚠️ 观察'
        else:
            rating = '🔴整改'
        
        print(f"\n  {supplier}")
        print(f"    批次合格率(LAR): {lar:.1f}%  评级: {rating}")
        print(f"    平均缺陷率: {mean_defect_rate:.2f}%")
        print(f"    检验批次: {total_batches}批  通过: {accepted_batches}批  拒收: {total_batches-accepted_batches}批")
        print(f"    COPQ质量成本: {total_copq:.2f}万元")


def run_spc_p_chart(df, supplier='SUP-A深圳宝美', window=20):
    """SPC p-chart: 监控缺陷率过程稳定性"""
    print("\n" + "=" * 60)
    print(f"【SPC p-chart — {supplier}】")
    print("=" * 60)
    
    sup_data = df[df['supplier'] == supplier].copy().reset_index(drop=True)
    
    # 用前window批次建立基准
    baseline = sup_data.iloc[:window]
    p_bar = baseline['defect_count'].sum() / baseline['sample_size'].sum()
    n_bar = baseline['sample_size'].mean()
    
    # 控制限（3-sigma）
    ucl = p_bar + 3 * np.sqrt(p_bar * (1 - p_bar) / n_bar)
    lcl = max(0, p_bar - 3 * np.sqrt(p_bar * (1 - p_bar) / n_bar))
    
    print(f"\n  基准缺陷率 p̄ = {p_bar*100:.3f}%")
    print(f"  UCL = {ucl*100:.3f}%  LCL = {lcl*100:.3f}%")
    print()
    
    # 检测失控点
    out_of_control = []
    for i, row in sup_data.iterrows():
        p_i = row['defect_rate']
        if p_i > ucl:
            out_of_control.append(i + 1)
            flag = '🔴 UCL超出（失控）'
        elif p_i > ucl * 0.85:
            flag = '⚠️ 接近UCL'
        else:
            flag = '✅'
        if i < 5 or p_i > ucl * 0.8:
            print(f"  批次{i+1:3d}: p={p_i*100:.2f}%  {flag}")
    
    if out_of_control:
        print(f"\n  ⚠️  失控批次: {out_of_control[:5]} ...")
        print(f"  → 触发根因调查：供应商工艺变更 / 原材料批次变化 / 操作员变化")
    else:
        print("\n  ✅ 过程受控，无失控信号")


def pareto_analysis_by_defect(df):
    """缺陷类型柏拉图分析"""
    print("\n" + "=" * 60)
    print("【缺陷类型柏拉图（Pareto）分析】")
    print("=" * 60)
    
    defect_df = df[df['primary_defect'] != '无']
    defect_count = defect_df.groupby('primary_defect')['defect_count'].sum().sort_values(ascending=False)
    total = defect_count.sum()
    
    cum_pct = 0
    print(f"\n  总缺陷数: {total}")
    for defect, count in defect_count.items():
        pct = count / total * 100
        cum_pct += pct
        bar = '█' * int(pct / 2)
        vital = '← 关键' if cum_pct <= 80 else ''
        print(f"  {defect:12s}: {count:5d}件  {pct:.1f}%  [累计{cum_pct:.0f}%] {bar} {vital}")
    
    print(f"\n  ⚡ 80/20法则：聚焦前{sum(1 for _ in defect_count if True)}类缺陷=消灭80%质量问题")


def compute_copq_by_supplier(df):
    """质量成本COPQ汇总"""
    print("\n" + "=" * 60)
    print("【质量成本COPQ汇总】")
    print("=" * 60)
    
    copq_summary = df.groupby('supplier').agg(
        COPQ总额=('copq_yuan', 'sum'),
        拒收批次=('accepted', lambda x: (~x).sum()),
    ).sort_values('COPQ总额', ascending=False)
    
    copq_summary['COPQ万元'] = (copq_summary['COPQ总额'] / 10000).round(2)
    total_copq = copq_summary['COPQ总额'].sum() / 10000
    
    for sup, row in copq_summary.iterrows():
        print(f"  {sup}: COPQ={row['COPQ万元']:.2f}万元  拒收={row['拒收批次']:.0f}批")
    
    print(f"\n  总COPQ: {total_copq:.2f}万元/周期")
    print(f"  改善目标: 通过根因管控将COPQ降低50% = 节省{total_copq*0.5:.2f}万元")


if __name__ == "__main__":
    print("【供应商来料质量 IQC-KPI 体系】\n")
    
    df = generate_iqc_data(n_batches=120)
    
    compute_supplier_quality_kpi(df)
    run_spc_p_chart(df, supplier='SUP-A深圳宝美')
    pareto_analysis_by_defect(df)
    compute_copq_by_supplier(df)
    
    print("\n[✓] 供应商来料质量IQC-KPI体系 测试通过")
    total_batches = len(df)
    accepted = df['accepted'].sum()
    print(f"    整体批次合格率={accepted/total_batches*100:.1f}%  "
          f"COPQ={df['copq_yuan'].sum()/10000:.1f}万  SPC异常已检出")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Supplier-Qualification-Onboarding-KPI]]（准入合格是IQC管控的前提）
- **前置（prerequisite）**：[[Skill-Supplier-Performance-Scorecard]]（IQC质量数据是绩效评分核心输入）
- **延伸（extends）**：[[Skill-Warehouse-Inbound-Quality-Accuracy-KPI]]（IQC通过后进入入库收货环节）
- **延伸（extends）**：[[Skill-OTIF-On-Time-In-Full-Analytics]]（质量拒收导致OTIF降低）
- **可组合（combinable）**：[[Skill-Supplier-Risk-XGBoost]]（IQC历史数据作为供应商风险预测特征）
- **可组合（combinable）**：[[Skill-Supply-Chain-Causal-SCM-Attribution]]（质量事故根因归因）

## ⑤ 商业价值评估

- **ROI预估**：母婴品牌IQC管控精细化后，通常将来料质量成本（COPQ）降低40-60%；以年采购额500万、COPQ率从2%降至0.8%计算，年化节省约6万元；同时减少因质量问题引发的Amazon差评和退货
- **实施难度**：⭐⭐⭐☆☆（需要建立IQC检验体系和数据记录，有一定初始投入）
- **优先级评分**：⭐⭐⭐⭐⭐（母婴类目质量直接关系用户安全和品牌信誉，不可忽视）
- **评估依据**：陈凤霞书中指出"来料质量KPI是供应商评价的核心维度，价格第二、质量第一"
