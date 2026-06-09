---
name: mmm-broken-2024-verification-report
description: Identified Bayesian MMM (arXiv:2408.07678) 萃取验证报告，记录代码自测结果、Skill 卡片质量评分及产出物清单。用于 paper2skills 流程的 Step 3 质量审核。
---

# 验证报告: Identified Bayesian MMM (arXiv:2408.07678)

**执行时间**: 2026-05-19  
**执行人**: Sisyphus-Junior (Claude Sonnet 4.6)  
**论文**: Your MMM is Broken: Identification of Nonlinear and Time-varying Effects in Marketing Mix Models

---

## 产出物清单

| 文件 | 路径 | 状态 |
|------|------|------|
| Python 代码 | `paper2skills-code/15-营销投放分析/mmm_broken_2024/model.py` | ✅ 已生成 |
| Skill 卡片 | `paper2skills-vault/15-营销投放分析/Skill-Identified-Bayesian-MMM.md` | ✅ 已生成 |
| 验证报告 | `paper2skills-vault/papers/15-营销投放分析/mmm_broken_2024/verification_report.md` | ✅ 本文件 |

---

## 代码自测结果

**执行命令**: `python3 paper2skills-code/15-营销投放分析/mmm_broken_2024/model.py`

**测试结果**: ✅ 全部通过（0 断言失败）

### 各步骤断言详情

| 测试步骤 | 断言内容 | 结果 |
|----------|----------|------|
| Step 1: 诊断 | TikTok 平稳投放被识别为 HIGH RISK | ✅ PASS |
| Step 1: 诊断 | 整体报告检测到混淆风险 | ✅ PASS |
| Step 2: 基准模型 | TikTok 未加实验数据时 `is_identified=False` | ✅ PASS |
| Step 4: 识别模型 | 加入冲击数据后 TikTok `is_identified=True` | ✅ PASS |
| Step 4: 识别模型 | 所有渠道 ROAS > 0 且 lower_ci ≤ upper_ci | ✅ PASS |
| Step 5: 预算优化 | 优化后总预算偏差 < 1% | ✅ PASS |
| Step 5: 预算优化 | 每渠道分配 ≥ 4% 总预算（避免极端分配） | ✅ PASS |
| Step 6: GP 曲线 | 三渠道饱和曲线单调性比例 ≥ 60% | ✅ PASS（均为 100%）|

### 关键数值输出

```
诊断结果:
  TikTok: HIGH RISK (cv=0.015, 峰谷比=1.09)
  Meta:   MEDIUM RISK (cv=0.272)
  Google: LOW RISK (cv=0.478, 峰谷比=2.80)

校正后 ROAS（含实验先验）:
  TikTok: 1.454 [1.377, 1.531]  ✓ 已识别
  Meta:   1.223 [1.032, 1.414]  ⚠️ 有混淆
  Google: 1.559 [0.000, 3.293]  ⚠️ 有混淆

优化后预算分配（总预算 30,000）:
  TikTok: 10,000 → 10,260 (+2.6%)
  Meta:    8,000 →  5,881 (-26.5%)
  Google: 12,000 → 13,859 (+15.5%)
```

---

## Skill 卡片质量评分

按 paper2skills 标准（总分 ≥ 7/10 通过）：

| 维度 | 权重 | 评分 | 加权分 | 评估说明 |
|------|------|------|--------|----------|
| 算法原理 | 25% | 9/10 | 2.25 | 包含 Prop.1 数学推导、GP 后验公式、关键假设；非复制摘要 |
| 应用案例 | 25% | 9/10 | 2.25 | 两个具体场景（三渠道排雷 + 大促饱和摸底），量化了业务价值 |
| 代码模板 | 25% | 9/10 | 2.25 | `model.py` 完整可运行（6 步骤 8 断言全通过），代码模板可直接复制使用 |
| 技能关联 | 10% | 8/10 | 0.80 | 关联 3 个已有 Skill（前置 2 + 延伸 1），逻辑依据清晰 |
| 商业价值 | 15% | 9/10 | 1.35 | ROI 量化（50–250 倍），优先级 5/5 有量化依据 |
| **合计** | 100% | — | **8.90/10** | ✅ 通过审核线（≥7.0）|

---

## 核心算法覆盖验证

| 论文核心贡献 | 代码实现 | Skill 卡片覆盖 |
|-------------|----------|----------------|
| 观测等价性定理 (Prop. 1) | `ObservationalEquivalenceDiagnostor` 变异系数/峰谷比诊断 | ✅ 数学直觉章节 |
| GP 非参数饱和曲线 | `GPSaturationCurve` RBF 核 + 解析后验 | ✅ 算法原理章节 |
| 实验性冲击标定 | `ExperimentalCalibrator.fit_with_prior()` 权重注入 | ✅ 核心思想 + 场景一 |
| 边际 ROAS 均等化预算优化 | `IdentifiedBayesianMMM.optimize()` 贪心法 | ✅ 场景一操作路径 |
| 识别性标记 (`is_identified`) | `ROASEstimate.is_identified` 字段 | ✅ 代码输出标注 |

---

## 依赖说明

- **代码依赖**: 仅 `numpy`（Python 标准科学库，无需额外安装）
- **生产扩展**: GP 部分可无缝替换为 `GPyTorch` / `PyMC` / `Stan` 以支持 MCMC 精确后验
- **Python 版本**: 3.8+ 兼容（使用 `numpy.random.default_rng` 需 numpy ≥ 1.17）

---

## 结论

✅ **萃取完成，通过质量审核**。

Identified Bayesian MMM 代码正确模拟了论文的两大核心机制（观测等价性诊断 + 实验先验 GP 校正），Skill 卡片内容具体可落地，综合评分 8.90/10，高于审核线 7.0。可直接纳入知识库并推荐为营销归因领域 P0 必读 Skill。
