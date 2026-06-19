---
title: Agent 置信度决策门控 — 高置信自动执行，低置信升级人工，防止 AI 乱操作
doc_type: knowledge
module: 16-智能体工程
topic: agent-decision-confidence-threshold
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill Card: Agent 置信度决策门控

> **论文**：Calibrated Uncertainty for Language Model Agents: When to Act Autonomously vs. Escalate
> **arXiv**：2402.07927 | 2024 | **桥梁**: 智能体工程 ↔ 风控反欺诈 | **类型**: 工程基础

## ① 算法原理

解决「AI Agent 做了错误决策，把价格改成了竞品的十分之一，损失 $50,000」的业务问题。

Agent 的决策错误主要来自两类场景：**模型对不确定情况假装有把握**（过度自信）和**模型对简单情况反复确认**（过度谨慎）。根本原因是 LLM 输出的概率没有校准到真实置信度。

**解决方案：三层置信度门控**

1. **校准置信度（Platt Scaling）**：将模型原始输出概率 p 映射到校准后的置信度 p_cal，让「说 90% 的决策」真实准确率确实接近 90%
2. **Temperature Scaling**：对 LLM Softmax 输出除以温度参数 T（T > 1 降低过度自信，T < 1 使分布更极端）
3. **三档执行门控**：
   - p_cal ≥ 0.85 → 全自动执行，无需人工
   - 0.6 ≤ p_cal < 0.85 → 通知人工确认，30 分钟内不响应则执行
   - p_cal < 0.6 → 停止执行，立即升级人工处理

**核心洞察**：置信度门控让 Agent 「知道自己不知道」，减少高风险错误，同时不影响高置信度决策的自动化效率。

校准质量评估：**ECE（Expected Calibration Error）**，ECE < 0.05 认为校准良好。

## ② 母婴出海应用案例

**场景A：定价 Agent 置信度门控——防止错误调价**
- 业务问题：竞品爬虫数据错误（竞品促销 -80%），定价 Agent 跟价导致 ASIN 亏本卖出 200 单
- 数据要求：历史决策记录（含 Agent 置信度分数 + 实际结果）+ 当前决策上下文
- 部署方案：定价决策置信度 < 0.75 时暂停自动调价，发送告警给运营审核
- 预期产出：错误调价事故从 12 次/月 → 1-2 次/月，年化避免损失 **$68,000**

**场景B：库存补货 Agent 不确定情景识别**
- 业务问题：旺季预测置信度低的情况下，Agent 自动大量补货导致滞销积压 $30,000
- 数据要求：销售预测置信区间 + 历史预测准确率分布
- 部署方案：预测 MAPE > 35% 时补货决策降级为「人工复核模式」，只提建议不执行
- 预期产出：旺季滞销风险降低 60%，库存周转率提升 22%

## ③ 代码模板

```python
"""
Agent 置信度校准与三档执行门控
Platt Scaling + Temperature Scaling + 决策路由
"""
import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Dict


class ExecutionTier(Enum):
    AUTO = "auto"           # 全自动执行（p >= 0.85）
    NOTIFY = "notify"       # 通知确认（0.6 <= p < 0.85）
    ESCALATE = "escalate"   # 升级人工（p < 0.6）


@dataclass
class Decision:
    """Agent 决策"""
    decision_id: str
    action: str
    raw_confidence: float     # 模型原始置信度（未校准）
    calibrated_confidence: float = 0.0
    tier: ExecutionTier = ExecutionTier.ESCALATE
    executed: bool = False
    actual_outcome: float = 0.0  # 事后结果（1=正确，0=错误）


class PlattScaler:
    """Platt Scaling 置信度校准器"""
    
    def __init__(self, a: float = 1.5, b: float = -0.3):
        """
        sigmoid(a * x + b) 校准曲线
        a > 1: 压缩过度自信；a < 1: 放大保守估计
        需要用历史数据拟合 a, b
        """
        self.a = a
        self.b = b
    
    def calibrate(self, raw_prob: float) -> float:
        """将原始概率校准到实际置信度"""
        logit = math.log(raw_prob / max(1 - raw_prob, 1e-6))
        cal_logit = self.a * logit + self.b
        return 1 / (1 + math.exp(-cal_logit))
    
    @staticmethod
    def temperature_scale(logits: List[float], temperature: float = 1.5) -> List[float]:
        """Temperature Scaling：降低过度自信"""
        scaled = [l / temperature for l in logits]
        max_val = max(scaled)
        exp_vals = [math.exp(l - max_val) for l in scaled]
        total = sum(exp_vals)
        return [e / total for e in exp_vals]
    
    def fit_from_history(self, decisions: List[Decision]):
        """从历史决策中拟合校准参数（简化版梯度下降）"""
        if len(decisions) < 20:
            print("  ⚠️  历史数据不足（<20 条），使用默认参数")
            return
        
        # 简化：按分位数校准
        bins = [[] for _ in range(10)]
        for d in decisions:
            bin_idx = min(int(d.raw_confidence * 10), 9)
            bins[bin_idx].append(d.actual_outcome)
        
        overconfidence_sum = 0
        for i, bin_decisions in enumerate(bins):
            if not bin_decisions:
                continue
            avg_raw = (i + 0.5) / 10
            avg_actual = sum(bin_decisions) / len(bin_decisions)
            overconfidence_sum += avg_raw - avg_actual
        
        # 根据过度自信程度调整 a 参数
        avg_overconf = overconfidence_sum / sum(1 for b in bins if b)
        if avg_overconf > 0.1:
            self.a = min(self.a * 1.2, 3.0)
            print(f"  🔧 校准器更新：a={self.a:.2f}（检测到过度自信 {avg_overconf:.2f}）")


def compute_ece(decisions: List[Decision], n_bins: int = 10) -> float:
    """
    Expected Calibration Error（ECE）
    ECE < 0.05 = 校准良好；> 0.15 = 严重失准
    """
    bins = [[] for _ in range(n_bins)]
    for d in decisions:
        bin_idx = min(int(d.calibrated_confidence * n_bins), n_bins - 1)
        bins[bin_idx].append((d.calibrated_confidence, d.actual_outcome))
    
    ece = 0.0
    n = len(decisions)
    for bin_items in bins:
        if not bin_items:
            continue
        avg_conf = sum(c for c, _ in bin_items) / len(bin_items)
        avg_acc = sum(o for _, o in bin_items) / len(bin_items)
        ece += (len(bin_items) / n) * abs(avg_conf - avg_acc)
    
    return ece


class ConfidenceGateRouter:
    """三档置信度决策路由器"""
    
    def __init__(
        self, 
        auto_threshold: float = 0.85,
        notify_threshold: float = 0.60,
        scaler: PlattScaler = None
    ):
        self.auto_threshold = auto_threshold
        self.notify_threshold = notify_threshold
        self.scaler = scaler or PlattScaler()
        self.decisions: List[Decision] = []
        self.stats: Dict[str, int] = {"auto": 0, "notify": 0, "escalate": 0}
    
    def route(self, decision: Decision) -> Tuple[ExecutionTier, str]:
        """路由决策到对应执行层"""
        # 校准置信度
        decision.calibrated_confidence = self.scaler.calibrate(decision.raw_confidence)
        p = decision.calibrated_confidence
        
        if p >= self.auto_threshold:
            tier = ExecutionTier.AUTO
            reason = f"置信度 {p:.2f} ≥ {self.auto_threshold} → 自动执行"
            decision.executed = True
            self.stats["auto"] += 1
        elif p >= self.notify_threshold:
            tier = ExecutionTier.NOTIFY
            reason = f"置信度 {p:.2f} ∈ [{self.notify_threshold}, {self.auto_threshold}) → 通知确认"
            self.stats["notify"] += 1
        else:
            tier = ExecutionTier.ESCALATE
            reason = f"置信度 {p:.2f} < {self.notify_threshold} → 升级人工"
            self.stats["escalate"] += 1
        
        decision.tier = tier
        self.decisions.append(decision)
        return tier, reason
    
    def print_report(self):
        total = len(self.decisions)
        if total == 0:
            return
        
        ece = compute_ece([d for d in self.decisions if d.actual_outcome != -1])
        
        print(f"\n📊 置信度门控执行报告（共 {total} 个决策）")
        print("-" * 50)
        print(f"  🟢 自动执行：{self.stats['auto']} ({self.stats['auto']/total:.0%})")
        print(f"  🟡 通知确认：{self.stats['notify']} ({self.stats['notify']/total:.0%})")
        print(f"  🔴 升级人工：{self.stats['escalate']} ({self.stats['escalate']/total:.0%})")
        print(f"\n  ECE（校准误差）：{ece:.4f} {'✅ 良好' if ece < 0.05 else '⚠️ 需要重校准'}")


# 运行验证
if __name__ == "__main__":
    random.seed(42)
    
    print("=" * 55)
    print("🎯 Agent 置信度决策门控演示（定价 Agent 场景）")
    print("=" * 55)
    
    scaler = PlattScaler(a=1.5, b=-0.3)
    router = ConfidenceGateRouter(
        auto_threshold=0.85,
        notify_threshold=0.60,
        scaler=scaler
    )
    
    # 模拟 30 个定价决策
    decision_scenarios = [
        ("D001", "调价 +5%", 0.92),   # 高置信：竞品涨价，跟价
        ("D002", "调价 -15%", 0.45),  # 低置信：数据异常（竞品可能促销）
        ("D003", "调价 +3%", 0.88),
        ("D004", "调价 -8%", 0.71),
        ("D005", "调价 +20%", 0.35),  # 低置信：波动太大
        ("D006", "维持价格", 0.95),
        ("D007", "调价 -3%", 0.78),
        ("D008", "调价 +12%", 0.55),
        ("D009", "维持价格", 0.91),
        ("D010", "调价 -25%", 0.28),  # 极低置信：疑似数据错误
    ]
    
    print()
    for did, action, raw_conf in decision_scenarios:
        d = Decision(decision_id=did, action=action, raw_confidence=raw_conf)
        tier, reason = router.route(d)
        
        tier_icon = {"auto": "✅", "notify": "⏳", "escalate": "🚫"}[tier.value]
        print(f"  {tier_icon} [{did}] {action:<12} | 原始: {raw_conf:.2f} → 校准: {d.calibrated_confidence:.2f} | {tier.value.upper()}")
        
        # 模拟事后结果（高置信决策大概率正确）
        correct_prob = 0.3 + d.calibrated_confidence * 0.65
        d.actual_outcome = 1.0 if random.random() < correct_prob else 0.0
    
    router.print_report()
    
    # 验证关键性质
    auto_decisions = [d for d in router.decisions if d.tier == ExecutionTier.AUTO]
    escalate_decisions = [d for d in router.decisions if d.tier == ExecutionTier.ESCALATE]
    
    assert all(d.calibrated_confidence >= 0.85 for d in auto_decisions), "自动执行的决策校准置信度应 >= 0.85"
    assert all(d.calibrated_confidence < 0.60 for d in escalate_decisions), "升级的决策校准置信度应 < 0.60"
    assert router.stats["auto"] + router.stats["notify"] + router.stats["escalate"] == len(decision_scenarios)
    
    # 测试温度缩放
    logits = [2.0, 1.0, 0.5, 0.2]
    probs_t1 = PlattScaler.temperature_scale(logits, temperature=1.0)
    probs_t2 = PlattScaler.temperature_scale(logits, temperature=2.0)
    assert probs_t2[0] < probs_t1[0], "Temperature > 1 应降低最高概率（减少过度自信）"
    assert abs(sum(probs_t1) - 1.0) < 0.001, "概率之和应为 1"
    
    print("\n[✓] Agent 置信度决策门控 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Decision-Confidence-Calibration-SC]]（供应链场景下的置信度校准方法论）
- **前置（prerequisite）**：[[Skill-Agent-Safety-Guardrails]]（安全护栏是置信度门控的上层框架）
- **延伸（extends）**：[[Skill-Human-in-Loop-Approval-Gate-Tag]]（通知确认和升级人工的具体实现）
- **可组合（combinable）**：[[Skill-MAS-Ecommerce-Ops-Automation]]（置信度门控 + 运营自动化 → 安全可靠的无人值守运营）

## ⑤ 商业价值评估

- **ROI 预估**：母婴跨境卖家，部署置信度门控后：
  - 防止错误调价事故：从 12 次/月 → 1-2 次/月，年化避免损失 **$68,000**
  - 防止错误大量补货：滞销库存风险降低 60%，年化挽回 **$18,000-30,000**
  - 运营信任度：Agent 「乱操作」投诉消失，运营愿意授权更多决策给 Agent 自动执行，自动化率提升 40%
- **实施难度**：⭐⭐☆☆☆（在现有 Agent 决策节点插入校准层即可，工程改动小）
- **优先级**：⭐⭐⭐⭐⭐（Agent 规模化部署的安全前提，缺少此机制容易发生灾难性错误）
- **特别建议**：新 Agent 上线前 2 周用「只通知不执行」模式积累校准数据，再开自动执行
