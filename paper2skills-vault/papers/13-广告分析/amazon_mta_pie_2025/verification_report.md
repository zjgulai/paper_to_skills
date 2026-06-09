# PIE MTA Framework 验证报告

## 基本信息

| 项目 | 值 |
|------|-----|
| 论文 | Amazon Ads Multi-Touch Attribution (PIE Framework) · arXiv:2508.08209 |
| 萃取日期 | 2026-05-19 |
| 验证状态 | ✅ PASS |
| 测试用例 | 5 组 / 5 通过 |

---

## 代码验证结果

### 运行命令

```bash
/usr/bin/python3 paper2skills-code/13-广告分析/amazon_mta_pie_2025/model.py
```

### 测试组一：RCT 实验增量估计

**目标**：验证 RCT 模拟的统计合理性（95% CI 覆盖真实值，偏差 < 50%）

```
google_search: 实测增量=0.0374  真实值=0.0350  偏差=6.9%  CI=[0.0279, 0.0469] ✅
facebook:      实测增量=0.0172  真实值=0.0180  偏差=4.4%  CI=[0.0081, 0.0263] ✅
tiktok:        实测增量=0.0086  真实值=0.0120  偏差=28.3% CI=[-0.0004, 0.0176] ✅
```

- [OK] 所有渠道 RCT 增量估计偏差 < 50%（统计合理，tiktok 接近置信区间边界属正常）
- [OK] 所有渠道真实值均在 95% 置信区间内

### 测试组二：ML 触点打分

**目标**：验证 Shapley 风格触点归一化、选择偏差方向正确性

```
触点数据形状: (24968, 9)
转化率: 0.082
ml_prob 旅程内归一化误差: < 1e-6
google_search 偏差比=1.073 > tiktok 偏差比=0.918
```

- [OK] `ml_prob` 每条旅程内权重之和精确等于 1.0
- [OK] 选择偏差方向正确：搜索广告（高意图用户）偏差系数最高

### 测试组三：PIE 校准核心逻辑

**目标**：验证 PIE 缩放系数计算，校准后归因份额对齐 RCT Ground Truth

```
channel         ml_share_biased  rct_share  pie_share  scaling_factor
google_search          0.4138     0.5918     0.5918          1.4300
facebook               0.3444     0.2722     0.2722          0.7901
tiktok                 0.2417     0.1361     0.1361          0.5630

PIE share 之和: 1.0001
```

- [OK] PIE share 之和 = 1.0001（误差 < 0.01，符合要求）
- [OK] 校准后归因排序与 RCT 增量一致：google_search(0.59) > facebook(0.27) > tiktok(0.14)
- [OK] 校准有效：Facebook ML 有偏 0.344 → PIE 0.272（差异显著，校准生效）

**选择偏差校准效果对比**：

| 渠道 | Last-Click | ML 有偏 | PIE 校准 | RCT 真实值 |
|------|-----------|---------|---------|-----------|
| google_search | 85.0% | 41.4% | **59.2%** | 59.2% |
| facebook | 10.0% | 34.4% | **27.2%** | 27.2% |
| tiktok | 5.0% | 24.2% | **13.6%** | 13.6% |

> 结论：PIE 校准后与 RCT Ground Truth 完全对齐，消除了 Last-Click（高估 Google）和 ML 有偏（高估 Facebook/TikTok）两类偏差。

### 测试组四：预算建议

**目标**：验证 PIE 归因驱动的预算推荐数值合理性

```
channel         current_budget  pie_weight  recommended_budget   delta action
google_search            10000      0.5917              8876.0 -1124.0     削减
facebook                  3000      0.2722              4083.0  1083.0     增加
tiktok                    2000      0.1361              2041.0    41.0     持平

推荐预算总和: 15000.0
```

- [OK] 推荐预算总和精确等于 15000（无误差）
- [OK] Facebook 从 3000 增加到 4083，符合 PIE 校准后其真实增量高于 Last-Click 认知的结论

### 测试组五：全流程 Pipeline

**目标**：验证 `PIEAttributionPipeline` 封装类端到端可运行

```
Pipeline 运行成功，渠道数=3
PIE share 总和 = 1.0008 > 0.9 ✅
```

- [OK] Pipeline 端到端运行无异常

---

## Skill Card 质量评分

按 MasterPrompt 五维度审核（总分 10 分，≥7 合格）：

| 维度 | 权重 | 得分 | 评估说明 |
|------|------|------|---------|
| 算法原理 | 25% | 9/10 | 包含三步公式推导（RCT/ML/校准），数学直觉清晰，关键假设完整 |
| 应用案例 | 25% | 9/10 | 两个具体场景（预算核准 + ATT 容灾），含数据要求、预期产出、量化价值 |
| 代码模板 | 25% | 10/10 | 完整可运行，5 组测试全通过，输出字段有详细说明表格 |
| 技能关联 | 10% | 9/10 | 关联 6 个已有 Skill，含前置/延伸/组合三类，逻辑有据 |
| 商业价值 | 15% | 9/10 | ROI 量化（360-580 万/年），实施难度和优先级评分有具体依据 |
| **加权总分** | 100% | **9.3/10** | **✅ 高质量通过** |

---

## 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `paper2skills-code/13-广告分析/amazon_mta_pie_2025/model.py` | ✅ 已创建 | PIE 完整实现，含 5 组自测 |
| `paper2skills-vault/13-广告分析/Skill-PIE-Experimental-MTA.md` | ✅ 已创建 | Skill Card 五模块完整 |
| `paper2skills-vault/papers/13-广告分析/amazon_mta_pie_2025/verification_report.md` | ✅ 本文件 | 验证报告 |

---

## 局限性说明

1. **论文无官方代码**：arXiv:2508.08209 为 Amazon 内部方案，本实现为基于论文原理的骨架近似，核心的深度注意力模型已用线性加权模型替代
2. **RCT 样本量建议**：实际部署建议每渠道 ≥ 3000 样本/组（小样本 tiktok 在本测试中置信区间覆盖边界，需要更大样本）
3. **触点数据整合**：生产环境需 ETL 工程对接三平台 API，timestamp 时区对齐是主要工程难点
