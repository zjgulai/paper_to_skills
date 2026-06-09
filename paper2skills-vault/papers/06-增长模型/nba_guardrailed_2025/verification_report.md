# 验证报告: Guardrailed CATE-NBA

**生成时间**: 2026-05-19  
**论文**: Guardrailed Uplift Targeting: A Causal Optimization Playbook for Marketing Strategy  
**arXiv ID**: 2512.19805

---

## 1. 代码自检结果

| 测试用例 | 描述 | 状态 |
|---------|------|------|
| 测试 1 | 基础流程：端到端流水线完整运行 + 返回类型正确 | ✅ 通过 |
| 测试 2 | 预算约束：总成本 ≤ 设定预算（500元 ≤ 500元）| ✅ 通过 |
| 测试 3 | 用户体验约束：每用户最多被分配 1 次行动 | ✅ 通过 |
| 测试 4 | 防食人化护栏：高净值用户的高额券 CATE 被打折 50% | ✅ 通过 |
| 测试 5 | 极端情况：零预算时分配结果为空（总成本=0, 分配数=0）| ✅ 通过 |
| 测试 6 | 大规模性能：10万用户 × 5行动，耗时 5.53s，触达 97,280 人 | ✅ 通过 |

**运行命令**: `/usr/bin/python3 paper2skills-code/06-增长模型/nba_guardrailed_2025/model.py`  
**最终输出**: `✅ 所有自检通过！Guardrailed CATE-NBA 模块验证完毕。`

---

## 2. Demo 场景跑批结果（5,000 用户 × 3 行动）

```
总触达用户数:  1,250
总营销成本:    10,000.0 元（= 预算上限）
预期总增量:    240.742
增量/成本比:   0.02407
行动分配明细:
  free_sample: 1,250 人（效价比最高：8元/次，增量显著）
  coupon_20:   0 人（在当前 mock 数据下，效价比低于 free_sample）
  coupon_50:   0 人（高成本 + 食人化惩罚后效价比最低）
```

**护栏生效记录**:
- 规则 1（食人化惩罚）：对 68 个高净值用户的 `coupon_50` CATE 打折 50%
- 规则 2（最低增量门槛）：过滤 374 个有效配对中 CATE < 0.02 的条目

---

## 3. 文件清单

| 文件 | 路径 | 状态 |
|------|------|------|
| Python 代码 | `paper2skills-code/06-增长模型/nba_guardrailed_2025/model.py` | ✅ 已创建 |
| Skill Card | `paper2skills-vault/06-增长模型/Skill-Guardrailed-CATE-NBA.md` | ✅ 已创建 |
| 萃取记录 | `paper2skills-vault/papers/06-增长模型/nba_guardrailed_2025/extract.md` | ✅ 已存在 |
| 验证报告 | `paper2skills-vault/papers/06-增长模型/nba_guardrailed_2025/verification_report.md` | ✅ 本文件 |

---

## 4. Skill Card 质量自评

| 维度 | 权重 | 得分 | 说明 |
|------|------|------|------|
| 算法原理 | 25% | 9/10 | 含 CATE 公式、护栏不等式、背包目标函数，非复制摘要 |
| 应用案例 | 25% | 8/10 | 2 个场景均为母婴出海具体痛点，含数据字段说明和量化价值 |
| 代码模板 | 25% | 9/10 | 完整可运行，含 6 个测试用例，含生产替换示例 |
| 技能关联 | 10% | 8/10 | 关联 4 个已有 Skill，含前置/延伸/组合三类 |
| 商业价值 | 15% | 8/10 | ROI 量化依据充分，难度/优先级评分有文献佐证 |
| **综合** | 100% | **8.6/10** | ≥ 7/10 质量基线，通过 ✅ |

---

## 5. 已知局限

1. **Mock CATE 简化**：当前 `MockCATEEstimator` 使用线性规则生成 CATE，无法反映真实的异质性处理效应。生产环境需替换为 `econml.CausalForestDML` 或 `causalml.uplift`。
2. **贪心近似**：`GreedyKnapsackPlanner` 对大规模问题（>100万用户）可能偏离精确解 1-5%。如需精确解，替换为 `PuLP` / `Google OR-Tools` 的 MIP 求解器。
3. **单期优化**：当前模型为静态单次分配，未考虑跨时间的序列决策（需扩展为 MDP / DQN 框架）。
