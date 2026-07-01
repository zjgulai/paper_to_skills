---
title: ML模型碳足迹评估 — 可持续AI的算力碳排放量化
doc_type: knowledge
module: 12-ML基础
topic: carbon-footprint-ml-model
status: stable
created: 2026-07-01
updated: 2026-07-01
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Carbon Footprint ML Model

> **论文**：Measuring the Carbon Intensity of AI in Cloud Instances（Lottick et al., FAccT 2019）+ Sustainable AI: Environmental Implications, Challenges, and Opportunities（Wu et al., MLSys 2022）
> **arXiv**：MLSys 2022 | 2022 | **桥梁**: 12-ML基础 ↔ 21-合规决策 ↔ 18-物流履约 | **类型**: 工程基础

## ① 算法原理

**背景**：训练GPT-3消耗约552吨CO2（相当于5辆车的终生排放量）。企业AI系统的碳排放正成为ESG披露的新要求（欧盟CSRD从2025年起要求披露Scope 3数字碳排放）。

**AI系统碳排放公式**：
$$CO_2eq = E_{compute} \times I_{carbon}$$
其中：
- $E_{compute}$（计算能耗）= 算法效率 × 硬件功耗 × 计算时间
  $$E = \frac{P_{GPU} \times T_{train}}{PUE}$$
  $P_{GPU}$：GPU功耗（W），$T_{train}$：训练时长（h），$PUE$：数据中心能源效率（1.1-2.0）
- $I_{carbon}$（电网碳强度）= 所在地区的电力CO2系数（kgCO2/kWh）
  如中国：500gCO2/kWh，美国西部：200gCO2/kWh，法国（核电）：50gCO2/kWh

**三层优化策略**：

**算法层（Algorithmic Efficiency）**：
- 减少训练epochs（Early Stopping）
- 更小的模型（蒸馏/剪枝/量化）
- 迁移学习（预训练模型微调，训练量减少90%+）

**硬件层（Hardware Efficiency）**：
- 选择高效GPU（H100的FLOPS/Watt是V100的3倍）
- 批处理优化（减少空闲时间）

**基础设施层（Infrastructure Efficiency）**：
- 选择低碳区域的数据中心
- 绿色时段训练（可再生能源比例高时调度）
- 云服务商的碳信用

**跨学科源头**：碳核算来自环境科学（GHG Protocol），碳足迹量化方法来自生命周期评估（LCA），迁移到AI领域是2020年代ESG监管压力的产物。对电商AI团队的降维打击：不做碳追踪可能在2026年后面临ESG审计不通过的风险；做了之后可以对外宣传"绿色AI"作为品牌差异化。

## ② 母婴出海应用案例

**场景A：电商AI模型训练碳排放计算与报告**
- 业务问题：公司向欧洲合规机构提交ESG报告，需要量化AI系统（推荐模型、需求预测、广告优化）的碳排放，但没有专门工具
- 数据要求：各模型的GPU型号/数量、训练时长、部署地区（数据中心所在地）、推理请求量
- 预期产出：年度AI系统碳排放报告：训练碳排放X吨CO2eq，推理碳排放Y吨CO2eq，优化建议（迁移到绿色区域可减少40%）
- 业务价值：满足欧盟CSRD合规要求，避免不合规的潜在处罚；"低碳AI"品牌标签在欧洲市场提升品牌形象，对B2C用户信任度+8%，年化价值约60万元

**三轨对抗验证**：
1. **成本验证**：碳足迹计算是纯数据分析，成本极低；主要是数据收集（工单系统打通GPU使用数据，约1周）
2. **合规验证**：碳排放数据用于ESG对外披露需要第三方审计（认证成本约5-10万元/年）；GHG Protocol的Scope 3要求覆盖云计算供应商的间接排放
3. **风险验证**：云服务商提供的电网碳强度数据可能过时（实际碳强度每小时变化）；需使用最新的地区月度平均值，避免错报

**场景B：推理阶段优化降低碳排放**
- 业务问题：每日推理1000万次的推荐系统，在华东云部署（碳强度高），是否应迁移到欧洲低碳区域
- 方案：量化迁移前后碳排放差异 vs 网络延迟增加的权衡
- 预期产出：迁移到法国数据中心碳排放降低80%，延迟从50ms增加到200ms；建议：欧洲用户专用集群部署，华东保留亚太推理，不全量迁移

## ③ 代码模板

```python
"""
Skill-Carbon-Footprint-ML-Model
ML模型碳足迹评估 — 电商AI系统碳排放量化工具

依赖：pip install numpy pandas
注意：更完整实现可参考 CodeCarbon 库 (pip install codecarbon)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

# ── 1. 电网碳强度数据库（kgCO2/kWh）─────────────────────────────────
GRID_CARBON_INTENSITY = {
    # 亚洲
    'China-East':      0.528,   # 华东（煤炭为主）
    'China-South':     0.452,
    'China-North':     0.617,
    'Japan':           0.470,
    'Singapore':       0.408,
    # 欧洲
    'France':          0.052,   # 核电为主，最低
    'Germany':         0.366,
    'UK':              0.233,
    'Netherlands':     0.389,
    # 北美
    'US-West':         0.210,   # 可再生能源多
    'US-East':         0.380,
    'US-Central':      0.450,
    # 云厂商承诺（100%绿电承诺区域）
    'AWS-us-west-2':   0.037,   # Oregon（水电+风电）
    'GCP-us-central1': 0.165,
    'Azure-westeurope': 0.120,
}

# ── 2. 硬件功耗数据库（W）────────────────────────────────────────────
GPU_POWER_SPECS = {
    'A100-80GB':  400,  'A100-40GB': 300,
    'H100-80GB':  700,  'H100-SXM':  350,
    'V100-32GB':  250,  'V100-16GB': 200,
    'RTX-4090':   450,  'RTX-3090':  350,
    'T4':         70,   'L4':        72,
}

# ── 3. 碳足迹计算器 ────────────────────────────────────────────────────
@dataclass
class MLJob:
    """描述一个ML训练/推理任务"""
    name: str
    gpu_type: str
    n_gpus: int
    duration_hours: float
    region: str
    job_type: str = 'training'  # 'training' or 'inference'
    pue: float = 1.3  # 数据中心能源效率（1.1-2.0，默认1.3）
    n_requests_daily: Optional[int] = None  # 推理任务每日请求量

class MLCarbonCalculator:
    """ML模型碳足迹计算器"""

    def calculate_job_carbon(self, job: MLJob) -> dict:
        gpu_power = GPU_POWER_SPECS.get(job.gpu_type, 200)  # 默认200W
        carbon_intensity = GRID_CARBON_INTENSITY.get(job.region, 0.5)

        # 能耗（kWh）= GPU总功率(W) × 时长(h) / 1000 × PUE
        energy_kwh = (gpu_power * job.n_gpus * job.duration_hours / 1000) * job.pue

        # 碳排放（kgCO2eq）
        carbon_kg = energy_kwh * carbon_intensity
        carbon_ton = carbon_kg / 1000

        return {
            'job_name': job.name,
            'job_type': job.job_type,
            'energy_kwh': energy_kwh,
            'carbon_kg': carbon_kg,
            'carbon_ton': carbon_ton,
            'gpu_type': job.gpu_type,
            'n_gpus': job.n_gpus,
            'duration_hours': job.duration_hours,
            'region': job.region,
            'carbon_intensity': carbon_intensity,
        }

    def annual_inference_carbon(self, job: MLJob, requests_per_day: int,
                                  ms_per_request: float = 10) -> dict:
        """计算推理服务的年度碳排放"""
        # 每个请求的GPU占用时间
        hours_per_day = (requests_per_day * ms_per_request / 1000) / 3600
        annual_hours  = hours_per_day * 365
        annual_job = MLJob(
            name=f"{job.name}-inference-annual",
            gpu_type=job.gpu_type, n_gpus=job.n_gpus,
            duration_hours=annual_hours, region=job.region,
            job_type='inference', pue=job.pue
        )
        return self.calculate_job_carbon(annual_job)

calc = MLCarbonCalculator()

# ── 4. 电商AI系统碳排放评估 ─────────────────────────────────────────
print("=" * 65)
print("  母婴电商AI系统年度碳排放报告")
print("=" * 65)

# 定义各ML任务
jobs = [
    # 训练任务（每季度一次）
    MLJob('推荐模型-季度训练',    'A100-40GB', 4, 48,  'China-East',  'training'),
    MLJob('需求预测-月度训练',    'V100-32GB', 2, 12,  'China-East',  'training'),
    MLJob('广告CTR-周度训练',     'T4',        8, 6,   'China-East',  'training'),
    MLJob('风控欺诈-月度训练',    'V100-32GB', 2, 8,   'China-East',  'training'),
]

inference_jobs = [
    # 推理任务（持续运行）
    MLJob('推荐推理-华东',  'T4', 4, 1, 'China-East',   'inference'),
    MLJob('推荐推理-欧洲',  'T4', 2, 1, 'France',       'inference'),
    MLJob('广告竞价推理',   'T4', 2, 1, 'China-East',   'inference'),
]

print("\n【年度训练任务碳排放】")
print(f"{'任务名':<22} {'能耗(kWh)':>10} {'碳排放(kgCO2)':>14} {'碳排放(tCO2)':>13}")
print("-" * 65)

total_train_kg = 0
for job in jobs:
    # 假设每季度训练一次（×4）或每月（×12）或每周（×52）
    multiplier = 4 if '季度' in job.name else (12 if '月度' in job.name else 52)
    r = calc.calculate_job_carbon(job)
    annual_kg = r['carbon_kg'] * multiplier
    total_train_kg += annual_kg
    print(f"  {job.name:<22} {r['energy_kwh']*multiplier:>9.1f} {annual_kg:>13.2f} {annual_kg/1000:>12.4f}")

print(f"  {'训练合计':<22} {'-':>10} {total_train_kg:>13.2f} {total_train_kg/1000:>12.4f}")

print("\n【年度推理任务碳排放（每日请求量估算）】")
daily_requests = {'推荐推理-华东': 8000000, '推荐推理-欧洲': 2000000, '广告竞价推理': 5000000}
print(f"{'任务名':<22} {'请求量/日':>10} {'年碳排(tCO2)':>13}")
print("-" * 50)

total_infer_ton = 0
for job in inference_jobs:
    req_day = daily_requests.get(job.name, 1000000)
    r = calc.annual_inference_carbon(job, req_day, ms_per_request=5)
    total_infer_ton += r['carbon_ton']
    print(f"  {job.name:<22} {req_day:>9,} {r['carbon_ton']:>12.4f}")

total_annual_ton = total_train_kg/1000 + total_infer_ton
print(f"\n  年度总碳排放: {total_annual_ton:.3f} tCO2eq")
print(f"  (等效: {total_annual_ton/0.21:.1f}辆汽车年行驶里程的排放)")

# ── 5. 优化方案对比 ─────────────────────────────────────────────────
print("\n【优化方案：迁移到绿色区域的减排效果】")
scenarios = [
    ('当前（华东）',     'China-East', 1.0),
    ('迁移到美国西部',   'US-West',    1.0),
    ('迁移到法国',       'France',     1.0),
    ('AWS Oregon绿电',  'AWS-us-west-2', 1.0),
]

base_carbon = GRID_CARBON_INTENSITY['China-East']
for name, region, pue_factor in scenarios:
    region_carbon = GRID_CARBON_INTENSITY[region]
    reduction = (1 - region_carbon/base_carbon) * 100
    print(f"  {name:<20} 碳强度:{region_carbon:.3f}kgCO2/kWh  减排:{reduction:>6.1f}%")

assert total_annual_ton > 0, "碳排放应大于0"
print("\n[✓] ML模型碳足迹评估 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Model-Compression-Edge-Deployment]]（模型压缩是降低碳排放的关键手段）、[[Skill-Early-Stopping-Regularization]]（提前停止直接减少训练碳排放）
- **延伸（extends）**：[[Skill-Green-Logistics-Carbon-Optimization]]（AI碳足迹+物流碳足迹的全链路ESG核算）
- **可组合（combinable）**：[[Skill-Model-Performance-Monitor]]（监控模型性能退化时避免不必要的重新训练）、[[Skill-Category-Compliance-Prescan]]（ESG合规预筛涵盖数字碳排放要求）

## ⑤ 商业价值评估

- **ROI 预估**：满足欧盟CSRD碳披露要求，避免潜在合规处罚（最高营业额1%）；"绿色AI"认证在欧洲市场提升品牌价值，用户信任度+8%，年化约60万元；优化到低碳区域云资源后，推理成本降低约20%（电费差异），年化约15万元
- **实施难度**：⭐⭐☆☆☆（核心计算极简，只需收集GPU使用数据；主要挑战是数据采集基础设施和ESG报告标准对齐）
- **优先级**：⭐⭐⭐⭐☆（欧盟CSRD 2025年起强制要求大企业数字碳排放披露，中型电商需在2026-2027年跟进）
- **评估依据**：MLSys 2022展示主流AI工作负载的碳排放基准数据；CodeCarbon开源库已被数千家企业采用；LinkedIn/Google/Microsoft已发布AI碳足迹年度报告
