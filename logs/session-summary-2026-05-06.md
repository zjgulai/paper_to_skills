# Session Summary 2026-05-06

## 主题
自迭代 LLM Agent 管线技能卡 — 完整工作流闭环（选题→萃取→审核→同步→提交→推送）

---

## 会话背景

用户希望萃取一个关于"autoresearch / 自迭代和自进化相关算法"的论文选题，业务场景聚焦：
- **A. 电商侧**：用自进化算法自动优化商品文案/定价/投放策略
- **C. 研究侧**：自动化竞品分析、市场情报自动萃取

经澄清和推荐，确定首选选题：**Self-Improving LLM Agent Pipeline（自迭代 LLM Agent 管线）**

**核心论文支撑**：
- The AI Scientist (arXiv:2408.06292, 2024)
- SEAL: Self-Adapting Language Models (NeurIPS 2025)
- Self-Challenging Language Model Agents (NeurIPS 2025)
- ETO: Exploration-Training Optimization (2024)

---

## 交付清单

### 1. 技能卡

| 属性 | 内容 |
|------|------|
| **名称** | Skill-Self-Improving-LLM-Agent-Pipeline |
| **路径** | `paper2skills-vault/07-NLP-VOC/Skill-Self-Improving-LLM-Agent-Pipeline.md` |
| **领域** | 07-NLP-VOC |
| **评分** | 8.0/10（审核通过） |
| **结构** | 5 模块完整：算法原理 + 业务案例 + 代码模板 + 技能关联 + 业务价值 |

**业务场景**：
- 场景 A：Momcozy 吸奶器 Amazon US 文案 CTR 自迭代优化（2.1% → 3.8%，+81%）
- 场景 C：竞品情报自动萃取准确率自提升（62% → 89%，+44%）

### 2. 代码模板

| 文件 | 路径 | 说明 |
|------|------|------|
| `model.py` | `paper2skills-code/nlp_voc/self_improving_llm_agent/` | 核心实现：ReflexionEngine + SelfRefineEngine + DPOTrainer + SelfImprovingAgent |
| `__init__.py` | 同上 | 模块导出 |
| `examples/openai_integration.py` | 同上/examples/ | OpenAI API 生产环境接入示例 |

**核心类**：
- `SelfImprovingAgent`：GRO 闭环基类（Generate → Review → Optimize）
- `CopyOptimizationAgent`：电商文案优化适配（metric_threshold=0.03）
- `IntelligenceExtractionAgent`：竞品情报萃取适配（metric_threshold=0.85）

### 3. 审核报告

| 属性 | 内容 |
|------|------|
| **路径** | `paper2skills-skills/paper-审核/reviews/review-Self-Improving-LLM-Agent-Pipeline-20260506.md` |
| **结论** | 通过（8.0/10） |
| **问题数** | 5 项（3 低 + 2 中优先级） |
| **修复状态** | 全部修复并验证通过 |

### 4. 同步状态

| 平台 | 状态 | 备注 |
|------|------|------|
| Vault | ✓ 已同步 | `paper2skills-vault/07-NLP-VOC/` |
| GitHub | ✓ 已同步 | `paper2skills-code/nlp_voc/self_improving_llm_agent/` |
| 飞书 | ✗ 未配置 | `~/.paper2skills/feishu_webhook` 不存在 |

---

## 工作流记录

```
选题 → 萃取 → 审核 → 同步 → 提交 → 推送
  ✓      ✓      ✓      ✓      ✓      ✓
```

**Commit**：`985e82b` feat: 新增自迭代 LLM Agent 管线技能卡
**Merge**：`b75b926` Merge branch 'main'（与远程 TJAP 修复合并）
**Push**：`main → origin/main` 成功

---

## 关键修复记录

### 修复 1：DPO 数据积累阈值过严

**问题**：`_accumulate_preference_data` 使用绝对差异阈值 `> 0.1`，对于 CTR 0.01-0.05 范围几乎不可能触发，导致测试中 DPO 偏好对为 0。

**修复**：改为复合阈值 `abs_diff > 0.005 or rel_diff > 0.1`

```python
# 修复前
if best.metric_value > worst.metric_value + 0.1:

# 修复后
abs_diff = best.metric_value - worst.metric_value
rel_diff = abs_diff / max(worst.metric_value, 1e-6)
if abs_diff > 0.005 or rel_diff > 0.1:
```

### 修复 2：DPO 偏好对去重

**问题**：同一 (context, winner, loser) 组合可被重复添加。

**修复**：`DPOTrainer` 新增 `_pair_keys` set + `_pair_key()` MD5 哈希去重。

### 修复 3：多 context 测试覆盖

**新增**：`test_dpo_accumulation()` — 3 个不同产品/画像组合各执行 5 轮，验证多 context 下 DPO 数据正常积累 + 去重机制有效。

### 修复 4：缺少生产环境接入示例

**新增**：`examples/openai_integration.py` — 完整演示 GPT-4o API 接入，含文案优化和情报萃取两个 Demo。

### 修复 5：技能卡内容补充

- DPO 公式后增加直觉解释（为什么偏好优化比监督学习更适合）
- 两个业务场景补充数据来源说明（Amazon Advertising API、人工抽检 20%）

---

## 技能关联

**前置技能**：
- Skill-LLM-Personalized-Marketing-Copy-Generation
- Skill-AutoTag-SelfEvolving-Label-System
- Skill-Review-Quality-Scoring

**可组合技能**：
- Skill-MAS-VOC-Data-Analyst
- Skill-AIPL-VOC-Lifecycle-Tags
- Skill-CrossLingual-Semantic-Alignment

---

## 遗留事项

1. **飞书同步未配置**：如需同步到飞书，需创建 `~/.paper2skills/feishu_webhook` 配置文件
2. **DPO 实际训练**：代码模板中的 DPOTrainer 仅管理数据，实际训练建议使用 TRL 库的 `DPOTrainer`
3. **评估 LLM 偏见**：审核报告提示，评估 LLM 的偏见会传导到策略更新，建议定期人工抽检反思质量

---

## 文件变更总览

```
paper2skills-code/nlp_voc/self_improving_llm_agent/
├── model.py                              [新增, 579行]
├── __init__.py                           [新增, 38行]
└── examples/
    └── openai_integration.py             [新增, 146行]

paper2skills-vault/07-NLP-VOC/
└── Skill-Self-Improving-LLM-Agent-Pipeline.md  [新增, 654行]

paper2skills-skills/paper-审核/reviews/
└── review-Self-Improving-LLM-Agent-Pipeline-20260506.md  [新增, 195行]

paper2skills-vault/07-资源库/
└── sync_status.json                      [更新, +6条记录]
```
