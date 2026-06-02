---
title: ProRCA — 因果图路径溯源根因分析
name: Skill-ProRCA-Business-Analysis
description: 基于因果依赖图与条件异常打分的端到端根因分析框架，将业务指标异动溯源时间从小时级压缩至秒级，适用于电商 GMV 暴跌等高损失场景。
module: data-agent-llm
topic: causal-root-cause-analysis
version: 0.1.0
status: stable
created: 2026-05-19
updated: 2026-05-19
paper: arXiv:2503.01475
source: ai
---

# ProRCA — 因果图路径溯源根因分析

## ① 算法原理

**核心问题**：当 GMV 暴跌时，传统监控会同时弹出几百个警报——流量跌、加购跌、结账跌、支付跌……却不告诉你**哪个是起因，哪个是被牵连的**。

**ProRCA 的三步流程**：

**Step 1：构建因果依赖图（Causal DAG）**
将业务指标之间的上下游关系显式建模为有向无环图（DAG）：
```
广告流量 → 加购量 → 结账到达量 → PayPal支付成功率 → GMV
```

**Step 2：条件异常打分（Conditional Anomaly Scoring）**
核心公式：
```
noise_i = z_i - β × avg(z_parents)
conditional_score = |noise_i|
```
- `z_i`：节点 i 当前的标准化异常值（z-score）
- `β`：上下游传导系数（典型值 0.7）
- 直觉：如果上游跌了，下游跟着跌是**正常传导**，不算二次异常。只有"给定父节点状态后，自身残差仍然爆表"的节点才获高分。

**Step 3：深度路径追踪（DFS Root Cause Tracing）**
从出问题的终点节点出发，逆着因果边，沿着**条件分数递增**的方向逐跳向上追溯，直到找到局部最高分节点，输出完整的因果链路径。

**关键假设**：
- 业务指标间的因果方向已知（可由系统拓扑或专家经验确定）
- 异常以 z-score 量化，传导为线性（β 可调）
- 真正的根因在给定父节点后的残差最大

---

## ② 母婴出海应用案例

### 场景 1：黑五大促 GMV 暴跌分钟级自动溯源

**业务问题**：黑五顶峰期每分钟 GMV 损失几十万美金，从告警触发到找出根因人工需要 1-2 小时。

**数据要求**：
- 各业务指标的 5 分钟滚动 z-score（流量、加购、转化、各支付通道成功率、GMV）
- 指标间因果拓扑（可从系统架构图自动提取或人工维护）

**实施步骤**：
1. GMV 跌破阈值触发告警
2. ProRCAEngine 加载实时指标 z-score 和因果图
3. `engine.analyze(trigger_node="GMV")` 秒级输出根因路径
4. LLM Agent 读取路径后生成自然语言报告推送到企业微信/钉钉

**预期产出**：
```
根因节点: PayPal支付成功率 (条件异常分数=3.03)
因果路径: PayPal支付成功率 → GMV
说明: 广告流量、加购量、信用卡通道均正常，
      北美区 PayPal 成功率从 97% 跌至 32%（下降 67%）是唯一真实根因。
```

**业务价值**：
- 诊断时间：从 60-120 分钟 → 30 秒内
- 大促 1 小时节省损失：按 GMV 暴跌 15%、每小时 GMV 100 万美金估算，可避免 15 万美金损失
- 额外价值：过滤 90%+ 的冗余警报，避免团队疲劳轰炸

---

### 场景 2：独立站多渠道转化率异动自动归因

**业务问题**：运营看到整体转化率下跌 0.5%，但不知道是哪个流量渠道、哪个支付链路、还是某品类货架出了问题。

**数据要求**：
- 按渠道（SEO / 付费 / 社媒）和支付方式（PayPal / Stripe / 本地支付）分维度的转化率 z-score
- 因果图：各渠道流量 → 分品类加购 → 各支付链路 → 整体转化率

**预期产出**：
- 自动缩小排查范围从 20+ 指标到 1-2 个真实异常节点
- 直接输出"问题链路"而非一张满是红色的仪表盘

**业务价值**：运营团队日均排查时间从 45 分钟 → 5 分钟，年化节省 180+ 人天。

---

## ③ 代码模板

> 完整可运行代码见：`paper2skills-code/09-DataAgent-LLM/prorca_2025/model.py`
> 运行方式：`python3 model.py`（纯标准库，无需额外依赖）

```python
from model import ProRCAEngine

# 1. 定义业务指标节点（z_score = 当前相对基线的标准化偏差）
nodes = [
    {"name": "广告流量",        "z_score":  0.2,  "raw_value": 28000, "raw_baseline": 27500},
    {"name": "加购量",          "z_score": -0.3,  "raw_value": 5100,  "raw_baseline": 5200},
    {"name": "结账到达量",      "z_score": -2.1,  "raw_value": 1850,  "raw_baseline": 2100},
    {"name": "PayPal支付成功率","z_score": -4.5,  "raw_value": 0.32,  "raw_baseline": 0.97},
    {"name": "信用卡支付成功率","z_score":  0.1,  "raw_value": 0.96,  "raw_baseline": 0.97},
    {"name": "GMV",             "z_score": -3.8,  "raw_value": 45000, "raw_baseline": 75000},
]

# 2. 定义因果边（parent → child 表示 parent 是 child 的原因）
edges = [
    ("广告流量", "加购量"),
    ("加购量", "结账到达量"),
    ("结账到达量", "PayPal支付成功率"),
    ("结账到达量", "信用卡支付成功率"),
    ("PayPal支付成功率", "GMV"),
    ("信用卡支付成功率", "GMV"),
]

# 3. 初始化引擎并分析
engine = ProRCAEngine(propagation_beta=0.7, score_threshold=0.5)
engine.load_graph(nodes, edges)
result = engine.analyze(trigger_node="GMV")

# 4. 输出人类可读报告
print(engine.summary(result))
# 输出:
# 根因节点   : PayPal支付成功率
# 追踪路径   : GMV → PayPal支付成功率
# 根因分数   : 3.030
# 因果链解释: 【根因】PayPal支付成功率(z=-4.50) → 【影响终点】GMV(z=-3.80)

# 5. 接入 LLM Agent（将 result 序列化传给大模型）
agent_prompt = f"""
根因分析完成。
根因节点: {result.root_cause}
因果路径: {' → '.join(result.path)}
解释: {result.explanation}

请用一句话向运营团队说明问题，并给出立即行动建议。
"""
```

**关键参数调优**：

| 参数 | 默认值 | 说明 | 调优方向 |
|------|--------|------|---------|
| `propagation_beta` | 0.7 | 上下游传导系数 | 指标间弱相关时调低至 0.4-0.5 |
| `score_threshold` | 0.5 | 追踪时父节点最低分数门槛 | 噪音大的系统调高至 0.8-1.0 |
| `max_depth` | 10 | 最大追溯跳数 | 微服务链路深时调高 |

---

## ④ 技能关联

**前置技能**：
- [[Skill-Root-Cause-Analysis-Agent]] — 了解 LLM 驱动的 RCA 框架后再引入因果图加持
- [[Skill-Argos-Agentic-Anomaly-Detection]] — 学会识别异常后，才需要溯源根因

**延伸技能**：
- [[Skill-SQL-Agent-Text-to-SQL]] — ProRCA 输出的根因路径可以驱动 SQL Agent 自动拉取佐证数据
- [[Skill-DeepAnalyze-Autonomous-Data-Science-Agent]] — 根因定位后交给 DeepAnalyze 做深度分析

**可组合**：
- **ProRCA + Argos**：Argos 负责实时发现哪个时间点异常，ProRCA 负责追问异常背后是什么原因
- **ProRCA + LLM Agent**：ProRCA 输出结构化路径，LLM 负责用自然语言汇报，实现"机器人分析师"

---

## ⑤ 商业价值评估

| 维度 | 评估 |
|------|------|
| **ROI 预估** | 大促场景：每次告警响应从 90 分钟→30 秒，按 GMV 损失率 15%、小时 GMV 100 万美金，单次可挽回 15 万美金。日常运营：运营团队每日节省 40 分钟排查，年化 160 人天。 |
| **实施难度** | ⭐⭐☆☆☆（2/5 星）— 纯 Python 无重依赖，最大工作量在维护因果图拓扑。 |
| **优先级评分** | ⭐⭐⭐⭐⭐（5/5 星）— 大促保障 P0 需求，直接与 GMV 损失挂钩，ROI 极高。 |
| **评估依据** | 因果图拓扑可从现有系统架构图半自动提取，初始建设成本约 3-5 人天。维护成本低（拓扑变更频率月级）。无外部 API 依赖，不存在数据安全风险。 |

---

*论文来源: arXiv:2503.01475 | 代码路径: `paper2skills-code/09-DataAgent-LLM/prorca_2025/model.py`*
