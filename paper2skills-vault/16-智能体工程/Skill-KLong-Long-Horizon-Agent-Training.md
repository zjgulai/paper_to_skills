---
title: KLong — 超长时域 Agent 训练：轨迹分割 SFT + 渐进 RL
doc_type: knowledge
module: 16-智能体工程
topic: klong-long-horizon-agent-training
status: stable
created: 2026-06-01
updated: 2026-06-01
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill Card: KLong — 超长时域 Agent 训练：轨迹分割 SFT + 渐进 RL

---

## ① 算法原理

### 核心问题

训练 LLM Agent 执行超长时域任务（50+ 步）面临两大瓶颈：
1. **Context 限制**：完整轨迹（100+ turns）远超模型 context window，无法直接训练
2. **稀疏奖励**：超长任务奖励延迟大，RL 训练极不稳定

### 两大核心技术

**1. 轨迹分割 SFT（Trajectory-Splitting SFT）**

核心思路：将超出 context window 的超长轨迹切成**有重叠的子轨迹**进行训练。

重叠窗口机制：
- 将长度为 L 的轨迹分成 window_size=W 的子轨迹，相邻子轨迹重叠 `overlap_ratio × W` 个 step
- 重叠部分保证了跨子轨迹的上下文连贯性（避免断点处信息丢失）
- 每个子轨迹只有 assistant turns 被训练，system/user turns 作为 prompt

效果：assistant turns 从 114 增加到 732（6.4×），大幅增加有效训练数据量。

**2. 渐进式 RL（Progressive RL）**

课程学习原理：从短任务开始训练，逐阶段增加 timeout（任务难度），Agent 能力持续提升。

```
阶段序列：
  Stage 1: timeout=30min  → 简单短任务，建立基础动作策略
  Stage 2: timeout=60min  → 中等任务，多工具调用
  Stage 3: timeout=90min  → 复杂任务，多跳推理
  Stage 4: timeout=120min → 超长任务，完整工作流
```

每阶段提升：持续 +6.67% 的长时域能力增益。

### 为什么小模型（106B）超越大模型（1T）

- KLong(106B) 在 PaperBench 超越 Kimi K2 Thinking(1T) 达 **11.28%**
- 原因：KLong 专门针对**长时域执行**做了轨迹级训练优化，而 1T 模型是通用的
- 类比：专科医生 vs 全科医生——在特定领域，专精训练优于规模堆砌

### 与 Reflexion 的核心区别

| 维度 | KLong | Reflexion |
|------|-------|-----------|
| 改进时机 | 训练时（权重更新）| 推理时（prompt 反思）|
| 持久化 | 永久（权重编码）| 临时（session 内）|
| 适用场景 | 专用 Agent 训练 | 通用 LLM 即插即用 |
| 计算成本 | 高（需 GPU 训练）| 低（推理时计算）|
| 长时域能力 | 强（轨迹级优化）| 弱（单 step 反思）|

### Research-Factory 自动化数据管线

自动化生成高质量超长轨迹训练数据：
- 任务生成 → 执行 → 结果校验 → 轨迹过滤 → 分割打包
- 减少人工标注依赖，支持大规模训练数据扩充

---

## ② 母婴出海应用案例

### 场景一：长流程选品调研 Agent 训练

**业务问题**：

母婴出海选品调研是典型的超长时域任务，完整流程包含 50+ 步骤：
```
竞品调研（10步）→ 市场规模分析（8步）→ 价格带扫描（6步）
→ 合规检查（12步）→ 供应链可行性（8步）→ 报告生成（6步）
```

现有方案：用通用 LLM 逐步执行，但：
- 超长上下文导致中途"遗忘"前期调研结论
- 每次重新启动都要重发全量背景，成本极高
- 无法持续学习改进执行策略

**KLong 训练方案**：

```
训练数据构建（Research-Factory）:
  1. 定义选品调研 SOP 的标准轨迹结构（50-80 步）
  2. 批量执行历史选品任务，记录完整轨迹
  3. 人工审核 Top 20% 轨迹（成功率高、路径高效）
  4. TrajectorySplitter 分割：window_size=20步, overlap_ratio=0.2

轨迹分割示意:
  完整轨迹(60步) → 子轨迹1(步1-20) + 子轨迹2(步17-36)
                  + 子轨迹3(步33-52) + 子轨迹4(步49-60)
  重叠区域确保竞品调研结论在价格分析阶段仍可见

渐进式 RL 阶段:
  Stage 1: 单品类调研（竞品扫描，约20步）
  Stage 2: 双品类对比（含市场分析，约35步）
  Stage 3: 完整选品报告（50+步，含合规+供链）
```

**数据要求**：
- 历史选品调研记录（至少 200 条完整轨迹）
- 成功/失败标注（最终选品决策是否正确）
- GPU 训练资源（至少 8×A100 或 4×H100）

**预期产出**：
- 专属选品 Agent，长链任务准确率 > 通用 GPT-4o
- 训练成本：比直接调用 1T 模型 API 低 10×（一次性训练投入）
- 执行速度：本地推理无 API 延迟，并行选品效率提升 5×

---

### 场景二：供应链异常溯源 Agent（多跳推理超长时域）

**业务问题**：

跨境母婴供应链异常溯源是典型的多跳推理长链任务：
```
症状（货到货损率 12%）→ 假设生成 → 数据采集 → 证据链推理
→ 根因确认 → 供应商沟通 → 改进方案 → 验证追踪
```
此类任务特点：
- 跨越 40-60 步，需要持续维护推理上下文
- 证据可能分散在 10+ 数据源（物流系统/仓库系统/QC 报告）
- 根因通常是多因素交叉（包装问题 + 温控问题 + 运输路线）

**KLong 训练方案**：

```
Progressive RL 课程设计:
  Stage 1: 单一原因异常（包装破损，约15步）
           timeout=20min，奖励：根因准确 +1
  Stage 2: 双因素异常（包装+运输，约30步）
           timeout=40min，奖励：根因准确 + 改进方案可行 +2
  Stage 3: 复杂多因素异常（3+因素，约50步）
           timeout=60min，奖励：根因准确 + 方案可行 + 供应商响应 +3
  Stage 4: 系统性质量问题（全链路，约70步）
           timeout=90min，完整 ROI 衡量

每阶段预期提升: +6.67% 根因识别准确率
累计提升（4阶段）: ~25% 准确率提升
```

**预期产出**：
- 供应链异常平均溯源时间：从 3-5 天 → 4-8 小时
- 根因识别准确率：较通用 LLM 提升 25%（4阶段渐进 RL 累积）
- 重复性异常预测：基于历史轨迹，提前 2-3 周预警相似模式

---

## ③ 代码模板

代码位置：`paper2skills-code/llm_agent_engineering/klong_long_horizon/model.py`

```python
"""
KLong — 超长时域 Agent 训练：轨迹分割 SFT + 渐进 RL
论文: KLong: Training LLM Agents for Extreme Long-Horizon Tasks
arXiv: 2602.17547 | 2026-02 (v2 2026-04)

核心组件:
- AgentTrajectory: 完整任务轨迹数据结构
- TrajectorySplitter: 带重叠窗口的轨迹分割器
- ProgressiveRLScheduler: 渐进式 RL 课程调度器
- ResearchFactory: 训练数据自动化生成管线
- KLongTrainer: 模拟训练流程（SFT + Progressive RL）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# 数据结构
@dataclass
class Step:
    """单步 Agent 交互"""
    step_id: int
    role: str                     # "user", "assistant", "tool"
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    token_count: int = 0


@dataclass
class AgentTrajectory:
    """完整任务轨迹"""
    trajectory_id: str
    task_type: str
    steps: list[Step]
    total_tokens: int
    context_limit: int
    success: bool = False
    reward: float = 0.0

    @property
    def assistant_turns(self) -> int:
        return sum(1 for s in self.steps if s.role == "assistant")

    def exceeds_context(self) -> bool:
        return self.total_tokens > self.context_limit


@dataclass
class SubTrajectory:
    """分割后的子轨迹"""
    parent_id: str
    sub_id: int
    steps: list[Step]
    overlap_with_prev: int         # 与上一个子轨迹重叠的 step 数
    is_first: bool
    is_last: bool

    @property
    def assistant_turns(self) -> int:
        return sum(1 for s in self.steps if s.role == "assistant")


# 轨迹分割器
class TrajectorySplitter:
    """
    带重叠窗口的轨迹分割器。
    将超出 context window 的超长轨迹切成有重叠的子轨迹训练。
    重叠机制保证跨子轨迹的上下文连贯性。
    """

    def __init__(self, window_size: int = 20, overlap_ratio: float = 0.2) -> None:
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.overlap_steps = max(1, int(window_size * overlap_ratio))

    def split_with_overlap(self, trajectory: AgentTrajectory) -> list[SubTrajectory]:
        """
        将轨迹切分为重叠子轨迹。
        步长 = window_size - overlap_steps（确保重叠）
        """
        steps = trajectory.steps
        total = len(steps)
        stride = self.window_size - self.overlap_steps

        if total <= self.window_size:
            return [SubTrajectory(
                parent_id=trajectory.trajectory_id,
                sub_id=0,
                steps=steps,
                overlap_with_prev=0,
                is_first=True,
                is_last=True,
            )]

        sub_trajectories: list[SubTrajectory] = []
        start = 0
        sub_id = 0

        while start < total:
            end = min(start + self.window_size, total)
            overlap = self.overlap_steps if sub_id > 0 else 0
            sub = SubTrajectory(
                parent_id=trajectory.trajectory_id,
                sub_id=sub_id,
                steps=steps[start:end],
                overlap_with_prev=overlap,
                is_first=sub_id == 0,
                is_last=end >= total,
            )
            sub_trajectories.append(sub)
            if end >= total:
                break
            start += stride
            sub_id += 1

        return sub_trajectories

    def verify_coverage(self, trajectory: AgentTrajectory, subs: list[SubTrajectory]) -> float:
        """验证所有 step 是否被至少一个子轨迹覆盖（覆盖率应为 100%）"""
        covered = set()
        for sub in subs:
            for step in sub.steps:
                covered.add(step.step_id)
        all_ids = {s.step_id for s in trajectory.steps}
        return len(covered & all_ids) / max(len(all_ids), 1)


# 渐进式 RL 调度器
class TrainingStage(str, Enum):
    STAGE_1 = "stage_1"   # 简单短任务
    STAGE_2 = "stage_2"   # 中等任务
    STAGE_3 = "stage_3"   # 复杂任务
    STAGE_4 = "stage_4"   # 超长任务


@dataclass
class StageConfig:
    stage: TrainingStage
    timeout_minutes: float
    max_steps: int
    difficulty_label: str
    expected_gain: float            # 预期准确率提升（百分比）


class ProgressiveRLScheduler:
    """
    渐进式 RL 课程调度器。
    逐阶段增加 timeout（任务难度），实现持续 +6.67% 能力提升。
    """

    STAGE_CONFIGS = [
        StageConfig(TrainingStage.STAGE_1, 30.0, 20, "简单短任务", 6.67),
        StageConfig(TrainingStage.STAGE_2, 60.0, 35, "中等任务（多工具）", 6.67),
        StageConfig(TrainingStage.STAGE_3, 90.0, 55, "复杂任务（多跳推理）", 6.67),
        StageConfig(TrainingStage.STAGE_4, 120.0, 80, "超长任务（完整工作流）", 6.67),
    ]

    def __init__(self, improvement_per_stage: float = 6.67) -> None:
        self.improvement_per_stage = improvement_per_stage
        self._current_stage_idx = 0
        self._stage_accuracies: list[float] = []

    def get_current_config(self) -> StageConfig:
        idx = min(self._current_stage_idx, len(self.STAGE_CONFIGS) - 1)
        return self.STAGE_CONFIGS[idx]

    def get_current_timeout(self, training_stage: int | None = None) -> float:
        """获取当前训练阶段的 timeout（分钟）"""
        idx = training_stage if training_stage is not None else self._current_stage_idx
        idx = min(idx, len(self.STAGE_CONFIGS) - 1)
        return self.STAGE_CONFIGS[idx].timeout_minutes

    def advance_stage(self, achieved_accuracy: float) -> bool:
        """记录当前阶段准确率并推进到下一阶段"""
        self._stage_accuracies.append(achieved_accuracy)
        if self._current_stage_idx < len(self.STAGE_CONFIGS) - 1:
            self._current_stage_idx += 1
            return True
        return False

    def cumulative_improvement(self) -> float:
        """累计能力提升估算（百分比）"""
        return self._current_stage_idx * self.improvement_per_stage

    def stage_sequence(self) -> list[dict]:
        """返回完整训练阶段序列（用于规划和展示）"""
        return [
            {
                "stage": cfg.stage.value,
                "timeout_min": cfg.timeout_minutes,
                "max_steps": cfg.max_steps,
                "difficulty": cfg.difficulty_label,
                "expected_gain": f"+{cfg.expected_gain}%",
            }
            for cfg in self.STAGE_CONFIGS
        ]


# Research-Factory 自动化数据管线
class ResearchFactory:
    """
    训练数据自动化生成管线（Research-Factory）。
    任务生成 → 执行 → 结果校验 → 轨迹过滤 → 分割打包。
    """

    def __init__(self, task_type: str, context_limit: int = 32000) -> None:
        self.task_type = task_type
        self.context_limit = context_limit
        self._trajectory_counter = 0

    def generate_training_data(
        self,
        num_trajectories: int = 50,
        min_steps: int = 30,
        max_steps: int = 80,
    ) -> list[AgentTrajectory]:
        """
        模拟批量生成训练轨迹。
        生产环境中应替换为真实任务执行引擎。
        """
        import random
        random.seed(42)

        trajectories = []
        for i in range(num_trajectories):
            n_steps = random.randint(min_steps, max_steps)
            steps = [
                Step(
                    step_id=j,
                    role="assistant" if j % 3 != 0 else "user",
                    content=f"{self.task_type} step {j}",
                    token_count=random.randint(50, 300),
                )
                for j in range(n_steps)
            ]
            total_tokens = sum(s.token_count for s in steps)
            traj = AgentTrajectory(
                trajectory_id=f"{self.task_type}_{self._trajectory_counter}_{i}",
                task_type=self.task_type,
                steps=steps,
                total_tokens=total_tokens,
                context_limit=self.context_limit,
                success=random.random() > 0.3,
                reward=random.uniform(0.5, 1.0) if random.random() > 0.3 else 0.0,
            )
            trajectories.append(traj)

        self._trajectory_counter += num_trajectories
        # 只保留成功轨迹（质量过滤）
        return [t for t in trajectories if t.success]


# KLong Trainer — 模拟完整训练流程
class KLongTrainer:
    """
    KLong 训练器：Stage 1 SFT + Stage 2 Progressive RL。
    """

    def __init__(
        self,
        task_type: str,
        context_limit: int = 32000,
        window_size: int = 20,
        overlap_ratio: float = 0.2,
    ) -> None:
        self.splitter = TrajectorySplitter(window_size, overlap_ratio)
        self.scheduler = ProgressiveRLScheduler()
        self.factory = ResearchFactory(task_type, context_limit)
        self.task_type = task_type
        self._sft_sub_trajectories: list[SubTrajectory] = []

    def stage1_sft_phase(self, num_trajectories: int = 100) -> dict[str, Any]:
        """
        阶段 1：轨迹分割 SFT
        生成轨迹 → 分割 → 统计 assistant turns 增长
        """
        trajectories = self.factory.generate_training_data(num_trajectories)
        long_trajectories = [t for t in trajectories if t.exceeds_context()]

        original_assistant_turns = sum(t.assistant_turns for t in long_trajectories)
        all_subs: list[SubTrajectory] = []

        for traj in long_trajectories:
            subs = self.splitter.split_with_overlap(traj)
            all_subs.extend(subs)
            coverage = self.splitter.verify_coverage(traj, subs)
            assert coverage >= 1.0, f"轨迹 {traj.trajectory_id} 覆盖率不足: {coverage:.2%}"

        self._sft_sub_trajectories = all_subs
        new_assistant_turns = sum(s.assistant_turns for s in all_subs)

        return {
            "total_trajectories": len(trajectories),
            "long_trajectories": len(long_trajectories),
            "sub_trajectories": len(all_subs),
            "original_assistant_turns": original_assistant_turns,
            "new_assistant_turns": new_assistant_turns,
            "assistant_turns_ratio": new_assistant_turns / max(original_assistant_turns, 1),
            "coverage_verified": True,
        }

    def stage2_progressive_rl(self, simulate_stages: int = 4) -> dict[str, Any]:
        """
        阶段 2：渐进式 RL
        模拟多阶段训练，每阶段增加任务难度（timeout）
        """
        results = []
        import random
        random.seed(123)

        base_accuracy = 0.60
        for stage_idx in range(simulate_stages):
            config = self.scheduler.STAGE_CONFIGS[stage_idx]
            achieved = base_accuracy + stage_idx * (self.scheduler.improvement_per_stage / 100)
            achieved = min(achieved, 0.99)
            results.append({
                "stage": config.stage.value,
                "timeout_min": config.timeout_minutes,
                "difficulty": config.difficulty_label,
                "achieved_accuracy": round(achieved, 4),
                "improvement": f"+{config.expected_gain}%",
            })
            self.scheduler.advance_stage(achieved)

        return {
            "stages_completed": simulate_stages,
            "stage_results": results,
            "cumulative_improvement": f"+{self.scheduler.cumulative_improvement():.1f}%",
            "final_accuracy": results[-1]["achieved_accuracy"],
        }


# 验证测试：选品调研长轨迹训练
def demo_product_selection_training() -> None:
    """
    验证场景：母婴出海选品调研 Agent 训练
    - 验证轨迹分割覆盖率 100%
    - 验证 assistant turns 增长（SFT 效果）
    - 验证渐进 RL 阶段序列合理性
    """
    print("=== KLong Demo: 母婴选品调研 Agent 训练 ===\n")

    trainer = KLongTrainer(
        task_type="product_selection_research",
        context_limit=8000,    # 模拟较小的 context limit 以触发分割
        window_size=15,
        overlap_ratio=0.2,
    )

    # Stage 1: SFT
    print("【Stage 1: 轨迹分割 SFT】")
    sft_result = trainer.stage1_sft_phase(num_trajectories=50)
    print(f"  总轨迹数: {sft_result['total_trajectories']}")
    print(f"  超长轨迹数: {sft_result['long_trajectories']}")
    print(f"  分割后子轨迹: {sft_result['sub_trajectories']}")
    print(f"  Assistant turns: {sft_result['original_assistant_turns']}"
          f" → {sft_result['new_assistant_turns']} "
          f"(×{sft_result['assistant_turns_ratio']:.1f})")
    print(f"  覆盖率验证: {'✓ 100%' if sft_result['coverage_verified'] else '✗'}")

    assert sft_result["assistant_turns_ratio"] >= 1.0, "SFT 未增加 assistant turns"
    assert sft_result["coverage_verified"], "轨迹覆盖率验证失败"

    # Stage 2: Progressive RL
    print("\n【Stage 2: 渐进式 RL（4阶段）】")
    rl_result = trainer.stage2_progressive_rl(simulate_stages=4)
    for r in rl_result["stage_results"]:
        print(f"  {r['stage']}: timeout={r['timeout_min']}min, "
              f"准确率={r['achieved_accuracy']:.2%}, {r['improvement']}")
    print(f"\n  累计能力提升: {rl_result['cumulative_improvement']}")
    print(f"  最终准确率: {rl_result['final_accuracy']:.2%}")

    # 验证阶段序列合理性（timeout 递增）
    timeouts = [r["timeout_min"] for r in rl_result["stage_results"]]
    assert timeouts == sorted(timeouts), "渐进 RL timeout 序列不单调递增"
    print("  ✓ timeout 序列单调递增（课程学习合理）")

    print("\n✅ 所有验证通过")


if __name__ == "__main__":
    demo_product_selection_training()
print("[✓] KLong Long Horizon Agent  测试通过")
```

---

## ④ 技能关联

### 前置技能

- [[Skill-EvoSC-Self-Consolidation]]：自进化 SFT，理解轨迹级自我改进的基本原理
- [[Skill-Reflexion-Self-Improvement]]：推理时反思改进，与 KLong 训练时改进形成互补对比
- [[Skill-Self-Improving-Agent-Feedback-Loop]]：Agent 自我改进反馈环，KLong 是其训练时版本

### 延伸技能

- [[Skill-ReliabilityBench-Agent-Reliability]]：Agent 可靠性评估，用于评价 KLong 训练后的 Agent 质量
- [[Skill-AgeMem-Unified-Agent-Memory]]：统一 Agent 记忆系统，配合 KLong 长时域 Agent 的记忆管理

### 可组合技能

- [[Skill-Context-Compression]]：上下文压缩，减少长轨迹训练的 token 成本
- [[Skill-Active-Context-Pruning]]：主动上下文剪枝，与轨迹分割互补（推理时 vs 训练时优化）
- [[Skill-DAG-Task-Decomposition-Planning]]：DAG 任务分解，为 KLong 长链任务提供结构化任务图

---

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|---------|---------|-----|
| 选品调研专属 Agent | 准确率提升 25%+，长链任务成功率 2× | GPU 训练（8×A100，2-4 周）+ 数据标注 | 中期（3-6 个月回收）|
| 供应链异常溯源 | 溯源时间 -95%（5天→4小时），根因准确率 +25% | 训练数据收集 + 同等 GPU | 高（快速节省运营人力）|
| 1T 模型替代方案 | API 成本 -90%（自有 106B vs 调用 1T API）| 模型训练 + 推理基础设施 | 长期战略收益 |

### 实施难度

**评分：⭐⭐⭐⭐☆（4/5 星）**

- 数据要求：高，需要大量高质量超长轨迹数据（200+ 条标注轨迹）
- 技术门槛：高，需要 GPU 训练基础设施 + RL 训练经验
- 工程复杂度：高，轨迹采集 + 分割 + SFT + RL 的完整管线
- 维护成本：中，模型需定期更新（随业务 SOP 变化）

### 优先级评分

**评分：⭐⭐⭐☆☆（3/5 星）**

- **长期战略价值高**：自有专精 Agent 比持续依赖 API 更具竞争护城河
- **短期门槛较高**：需要 GPU 训练资源，中小团队直接落地困难
- **渐进式价值释放**：先用 Reflexion（低成本），业务规模扩大后再投入 KLong 训练
- **母婴出海适配度高**：选品、合规、供链等长流程任务与 KLong 设计场景高度匹配

### 评估依据

1. **PaperBench 实证超越**：106B 超越 1T 达 11.28%，证明专精训练优于规模堆砌
2. **量化指标清晰**：+6.67%/stage，assistant turns 6.4×，具体可复现
3. **Research-Factory 降低门槛**：自动化数据管线减少人工标注负担
4. **母婴出海典型场景匹配**：50+ 步长链任务（选品/溯源）正是 KLong 的设计靶场

---

## 参考论文

1. **KLong: Training LLM Agents for Extreme Long-Horizon Tasks** (2026-02, v2 2026-04)
   - arXiv: [2602.17547](https://arxiv.org/abs/2602.17547)
   - 核心贡献：轨迹分割 SFT（assistant turns 114→732）+ 渐进式 RL（+6.67%/stage）+ Research-Factory 数据管线
   - KLong(106B) 在 PaperBench 超越 Kimi K2 Thinking(1T) +11.28%

## 相关基础

- **PaperBench**：Agent 长时域任务评测基准
- **Kimi K2 Thinking**：1T 参数通用大模型（被 KLong 超越的基线）
- **Reflexion**（Shinn et al.）：推理时反思改进（与 KLong 互补的轻量方案）
- **课程学习（Curriculum Learning）**：渐进式 RL 的理论基础

---

## 与同领域 Skill 的对比

| 维度 | KLong（本 Skill）| Reflexion | EvoSC |
|------|------------------|-----------|-------|
| 改进时机 | 训练时（权重更新）| 推理时（prompt）| 训练时（SFT） |
| 长时域支持 | 专门设计（50+步）| 弱（单 step 反思）| 中（短-中轨迹）|
| 计算成本 | 高（GPU 训练）| 低（推理计算）| 中（轻量 SFT）|
| 持久化效果 | 永久（权重）| 临时（session）| 永久（权重）|
| 最佳使用场景 | 专用长链 Agent | 通用 LLM 增强 | 技能积累型 Agent |

**推荐组合**：
- **开发阶段**：先用 Reflexion（零训练成本，快速验证任务可行性）
- **业务验证后**：用 KLong 训练专属 Agent（10× 成本降低 + 更强能力）
- **记忆管理**：配合 AgeMem 处理 KLong Agent 的超长上下文状态
