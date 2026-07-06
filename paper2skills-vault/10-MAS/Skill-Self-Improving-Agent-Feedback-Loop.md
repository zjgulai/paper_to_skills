---
title: Self-Refine + RL — 反馈闭环与自进化 Agent
doc_type: knowledge
module: 10-MAS
topic: self-improving-agent-feedback-loop
status: stable
created: 2026-05-10
updated: 2026-05-10
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill: Self-Refine + RL — 反馈闭环与自进化 Agent

---

## ① 算法原理

### 核心思想

**Self-Refine** 是一种让 Agent 对自身输出进行批评和改进的迭代机制。核心洞察：**语言模型不仅能生成内容，也能评估和改进内容**——利用同一模型的双重能力，实现无需外部监督的自我进化。

Self-Refine 的四个步骤：
1. **Generate**：Agent 生成初始输出
2. **Feedback**：Agent 对自身输出进行批评（识别问题、遗漏、不一致）
3. **Refine**：Agent 基于批评改进输出
4. **Iterate**：重复 Feedback-Refine 直到满足质量阈值

**经验记忆库（Memory Bank）** 扩展了 Self-Refine：
- 将成功和失败的经验存入长期记忆
- 支持相似情况的经验检索和复用
- 通过成功率排序实现经验优先级管理

**反馈闭环编排器（Feedback Loop Orchestrator）** 将两者结合：
- 执行任务前检索相似经验
- 执行中使用 Self-Refinement 迭代改进
- 执行后收集反馈并存入记忆
- 形成持续进化的闭环

### 数学直觉

**Self-Refine 迭代**：

设生成函数为 $G$，反馈函数为 $F$，改进函数为 $R$。迭代过程：

$$o_0 = G(x)$$
$$f_t = F(o_t)$$
$$o_{t+1} = R(o_t, f_t)$$

其中 $o_t$ 是第 $t$ 轮输出，$f_t$ 是对 $o_t$ 的反馈。停止条件：质量 $Q(o_t) \geq \theta$ 或达到最大迭代次数 $T$。

**经验记忆检索**：

对于新任务 $x$，检索相似经验：

$$\text{score}(x, e_i) = \text{sim}(x, e_i^{\text{situation}}) \cdot e_i^{\text{success\_rate}}$$

选择 top-$k$ 经验作为上下文增强输入：

$$x' = x \oplus \{e_i | \text{score}(x, e_i) \text{ 在 top-}k\}$$

### 关键假设

1. **模型可自我评估**：同一模型既能生成也能批评
2. **反馈可转化为改进**：批评意见可以指导输出修正
3. **经验可复用**：相似任务的经验对未来任务有帮助
4. **质量可量化**：存在可靠的质量评估指标

---

## ② 母婴出海应用案例

### 场景一：VOC 分析 Agent 的持续进化

**业务问题**：

VOC 分析 Agent 在处理新类型评论时表现不稳定。例如新出现的品牌名、产品型号、方言表达可能导致实体识别失败。需要 Agent 能从错误中学习并自我改进。

**数据要求**：

- VOC 分析任务的执行轨迹（输入、输出、反馈）
- 人工评分（1-5 分）
- 成功/失败案例的标注

**预期产出**：

```
初始执行:
  输入: "Spectra S1 吸奶器非常好用，静音效果很好"
  输出: {entities: [{text: "吸奶器", type: "PRODUCT"}], sentiment: positive, confidence: 0.75}
  反馈: "未识别品牌名 Spectra S1，置信度过低"

Self-Refine 迭代 1:
  改进: 加入品牌名识别，更新实体为 "Spectra S1 吸奶器"
  质量: 0.82

迭代 2:
  改进: 识别 "静音" 为关键属性
  质量: 0.91 → 满足阈值，停止

经验存入记忆库:
  {situation: "吸奶器评论", action: "品牌名+属性识别", outcome: success, lesson: "静音是关键属性", success_rate: 0.92}

下次执行相似任务时:
  检索到经验 → 自动增强输入 → 首次输出质量即达 0.88
```

**业务价值**：
- Agent 准确率随使用次数提升（无需重新训练模型）
- 减少人工审核工作量 50-60%
- 新类型评论的处理能力自动增强

---

### 场景二：跨 Agent 经验共享

**业务问题**：

多个 VOC Agent 分别处理不同品类的评论（吸奶器、储奶袋、推车等），每个 Agent 独立积累经验，造成知识孤岛。

**数据要求**：

- 多 Agent 的执行轨迹
- 共享记忆库（向量数据库）
- 经验相似度计算模型

**预期产出**：

```
Agent A（吸奶器）经验:
  "静音" 是高频正面属性，"漏奶" 是高频负面属性

Agent B（储奶袋）执行新任务:
  检索共享记忆 → 发现 "密封性" 类似于 "静音"（正面属性）
  → 首次输出即正确识别 "密封性好" 为正面情感

跨 Agent 聚合统计:
  总执行次数: 10,000
  成功率: 78% → 89%（6个月后）
  平均迭代次数: 2.3 → 1.4（经验积累后首次质量提升）
  记忆库规模: 1,200 条经验
```

**业务价值**：
- 经验跨品类复用，加速新 Agent 冷启动
- 全局知识库持续积累，形成组织级 AI 资产
- 质量趋势可追踪，支持 ROI 量化

---

**三轨验证** | 成本轨：Agent协同调度API月均成本1200元（Claude API调用约50万tokens/月），人工反馈标注12小时/月，成本约3000元/月，总计4200元/月 | 合规轨：符合跨境电商数据合规要求，用户反馈数据本地存储不出境，满足个保法和平台政策 | 风险轨：自学习反馈循环可能导致模型漂移，建议每月进行准确率基准测试，异常波动超过3%时触发人工审查，预留重训练成本5000元/季度

**三轨验证** | 成本轨：多Agent并发调度月均API成本2000元（tokens消耗增加40%），人工验证大促期间增至30小时/月，成本约7500元/月，总计9500元/月 | 合规轨：大促期间数据处理量增加5倍，需确保符合跨境数据传输协议，建议部署本地缓存层降低出境数据量30% | 风险轨：Agent间反馈冲突概率上升至8%，可能导致备货决策震荡，建议设置反馈权重衰减机制，单Agent权重上限0.3，预留应急人工介入成本10000元/大促

## ③ 代码模板

代码位置：`paper2skills-code/mas/feedback_loop/self_improving_agent.py`

核心组件：
- `ExecutionTrace`: 执行轨迹记录
- `Experience`: 经验数据结构
- `MemoryBank`: 经验记忆库（添加、检索、去重、容量控制）
- `SelfRefinementEngine`: 自我反思引擎（Feedback-Refine-Iterate）
- `FeedbackLoopOrchestrator`: 反馈闭环编排器
  - `execute_with_feedback`: 完整闭环执行
  - `get_performance_stats`: 性能统计

运行方式：
```bash
cd paper2skills-code/mas/feedback_loop
python self_improving_agent.py
```

生产环境建议：
1. 使用向量数据库（Pinecone/Milvus）存储和检索经验
2. 实现 RLHF 循环：人类反馈 → 奖励模型 → 策略优化
3. 建立 A/B 测试框架对比不同策略效果
4. 设置质量门禁（Quality Gate）防止低质量输出流入生产
5. 定期清理和合并记忆库，防止知识陈旧

---

## ④ 技能关联

### 前置技能
- **AutoGen**：多 Agent 对话框架，提供 Agent 基础设施
- **MetaGPT**：SOP 驱动协作，提供结构化工作流
- **LLM 基础**：理解 Prompt Engineering、上下文学习

### 延伸技能
- **RLHF**：人类反馈强化学习，提升反馈信号质量
- **向量检索**：高效的经验相似度检索
- **A/B 测试**：策略效果对比框架

### 可组合技能
- **AutoGen**：Self-Refinement 可以作为 AutoGen Agent 的内部机制
- **MetaGPT**：经验记忆可以作为共享知识库注入 SOP 流程
- **语义蓝图编译器**：质量评估可以基于语义蓝图的结构化约束
- **可组合**：[[Skill-MAS-Orchestrator]] / [[Skill-ReAct-Reasoning-Acting]]

---

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|---------|---------|-----|
| Agent 自我进化 | 准确率从 75% → 90%+，减少人工干预 | 开发 2-3 周 | 15-20x |
| 经验共享复用 | 新 Agent 冷启动时间缩短 60% | 开发 1-2 周 | 10-15x |
| 质量趋势追踪 | 可量化的 AI 资产积累，支持决策 | 开发 1 周 | 8-12x |

### 实施难度
**评分：⭐⭐⭐⭐☆（4/5星）**

- 数据要求：中，需要执行轨迹和反馈数据
- 技术门槛：中高，需要理解 Self-Refine 和记忆检索机制
- 工程复杂度：中高，涉及多组件协调（生成、反馈、记忆、编排）
- 维护成本：中，记忆库需要定期清理和更新

### 优先级评分
**评分：⭐⭐⭐⭐⭐（5/5星）**

- **核心差异化**：Self-Improving Agent 是系统持续进化的关键
- **复利效应**：每次执行都在积累知识，长期价值巨大
- **技术前沿**：Self-Refine 是 2023 NeurIPS 热点方向
- **业务刚需**：VOC 分析需要持续适应新品牌、新表达、新场景

---

## 参考论文

1. **Self-Refine: Iterative Refinement with Self-Feedback** (NeurIPS 2023)
   - Madaan, A. et al. (CMU / AI2)
   - 核心贡献：无需外部监督，LLM 自我生成反馈并迭代改进
   - arXiv：2303.17651

2. **Reflexion: Language Agents with Verbal Reinforcement Learning** (NeurIPS 2023)
   - Shinn, N. et al. (Northeastern / MIT)
   - 核心贡献：将执行反馈转化为 verbal reinforcement，实现策略进化
   - arXiv：2303.11366

---

## 在 MAS 工作流中的位置

```
[MAS Orchestrator]
    ↓ 分发任务
[Subagent Decomposer]
    ↓ 分解为子任务
[Agent 1] → [Self-Refine] → [Memory Bank]
[Agent 2] → [Self-Refine] → [Memory Bank]
    ↓ 共享经验
[全局 Memory Bank]
    ↓ 检索相似经验
[下次执行] → 质量提升
    ↓
[反馈/评估/记忆/再训练] → loop back
```

---

## ⑥ 可运行代码实现

```python
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import math
import random

# ============================================================
# 1. 数据结构定义
# ============================================================

class ExecutionTrace:
    """执行轨迹记录"""
    def __init__(self, task_input: str, initial_output: str, final_output: str,
                 feedbacks: List[str], quality_scores: List[float], success: bool):
        self.task_input = task_input
        self.initial_output = initial_output
        self.final_output = final_output
        self.feedbacks = feedbacks
        self.quality_scores = quality_scores
        self.success = success

class Experience:
    """经验数据结构"""
    def __init__(self, situation: str, action: str, outcome: str,
                 lesson: str, success_rate: float, embedding: Optional[np.ndarray] = None):
        self.situation = situation
        self.action = action
        self.outcome = outcome
        self.lesson = lesson
        self.success_rate = success_rate
        self.embedding = embedding if embedding is not None else np.random.randn(10)

    def __repr__(self):
        return f"Experience(situation='{self.situation}', action='{self.action}', success_rate={self.success_rate:.2f})"

# ============================================================
# 2. 记忆库 (Memory Bank)
# ============================================================

class MemoryBank:
    """经验记忆库：添加、检索、去重、容量控制"""
    def __init__(self, max_size: int = 100, similarity_threshold: float = 0.8):
        self.experiences: List[Experience] = []
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold

    def add_experience(self, exp: Experience) -> bool:
        """添加经验，自动去重和容量控制"""
        # 去重检查：基于 situation 相似度
        for existing in self.experiences:
            sim = self._cosine_similarity(exp.embedding, existing.embedding)
            if sim > self.similarity_threshold:
                # 更新已有经验的成功率（取平均）
                existing.success_rate = (existing.success_rate + exp.success_rate) / 2
                return False

        # 容量控制：如果超过最大容量，移除成功率最低的经验
        if len(self.experiences) >= self.max_size:
            min_idx = min(range(len(self.experiences)),
                         key=lambda i: self.experiences[i].success_rate)
            self.experiences.pop(min_idx)

        self.experiences.append(exp)
        # 按成功率排序
        self.experiences.sort(key=lambda e: e.success_rate, reverse=True)
        return True

    def retrieve_similar(self, query_embedding: np.ndarray, top_k: int = 3) -> List[Experience]:
        """检索相似经验"""
        if not self.experiences:
            return []

        scores = []
        for exp in self.experiences:
            sim = self._cosine_similarity(query_embedding, exp.embedding)
            score = sim * exp.success_rate  # 结合相似度和成功率
            scores.append((score, exp))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [exp for _, exp in scores[:top_k]]

    def get_stats(self) -> Dict:
        """获取记忆库统计信息"""
        if not self.experiences:
            return {"size": 0, "avg_success_rate": 0.0, "max_success_rate": 0.0, "min_success_rate": 0.0}

        success_rates = [e.success_rate for e in self.experiences]
        return {
            "size": len(self.experiences),
            "avg_success_rate": float(np.mean(success_rates)),
            "max_success_rate": float(max(success_rates)),
            "min_success_rate": float(min(success_rates))
        }

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

# ============================================================
# 3. 自我反思引擎 (Self-Refinement Engine)
# ============================================================

class SelfRefinementEngine:
    """自我反思引擎：模拟 Feedback-Refine-Iterate 过程"""
    def __init__(self, max_iterations: int = 5, quality_threshold: float = 0.85):
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold

    def execute(self, task_input: str, memory_bank: Optional[MemoryBank] = None) -> ExecutionTrace:
        """
        执行自我反思迭代
        模拟：生成初始输出 → 反馈 → 改进 → 迭代
        """
        # 模拟初始输出
        initial_output = self._generate_initial(task_input)

        # 检索相似经验（如果有记忆库）
        if memory_bank and memory_bank.experiences:
            query_emb = np.random.randn(10)  # 模拟查询嵌入
            similar_exps = memory_bank.retrieve_similar(query_emb, top_k=2)
            if similar_exps:
                # 利用经验增强初始输出
                initial_output += f" [增强: 参考经验 '{similar_exps[0].lesson}']"

        current_output = initial_output
        feedbacks = []
        quality_scores = []
        success = False

        for iteration in range(self.max_iterations):
            # 模拟反馈生成
            feedback = self._generate_feedback(current_output, task_input)
            feedbacks.append(feedback)

            # 模拟质量评分
            quality = self._evaluate_quality(current_output, task_input)
            quality_scores.append(quality)

            # 检查是否满足质量阈值
            if quality >= self.quality_threshold:
                success = True
                break

            # 模拟改进
            current_output = self._refine_output(current_output, feedback, task_input)

        return ExecutionTrace(
            task_input=task_input,
            initial_output=initial_output,
            final_output=current_output,
            feedbacks=feedbacks,
            quality_scores=quality_scores,
            success=success
        )

    def _generate_initial(self, task_input: str) -> str:
        """模拟初始生成"""
        # 简单模拟：提取关键词
        words = task_input.lower().split()
        entities = [w for w in words if len(w) > 2][:3]
        return f"分析结果: 识别到实体 {entities}, 情感: 正面, 置信度: 0.75"

    def _generate_feedback(self, output: str, task_input: str) -> str:
        """模拟反馈生成"""
        if "置信度" in output and "0.75" in output:
            return "反馈: 置信度偏低，建议提高识别精度"
        if "实体" in output and len(output) < 50:
            return "反馈: 输出过于简单，缺少详细属性分析"
        return "反馈: 输出质量良好，可进一步优化细节"

    def _evaluate_quality(self, output: str, task_input: str) -> float:
        """模拟质量评估"""
        base_quality = 0.7
        # 根据输出长度和内容丰富度调整
        if len(output) > 80:
            base_quality += 0.1
        if "置信度" in output:
            base_quality += 0.05
        if "属性" in output or "特征" in output:
            base_quality += 0.1
        return float(min(base_quality + random.uniform(-0.05, 0.05), 1.0))

    def _refine_output(self, output: str, feedback: str, task_input: str) -> str:
        """模拟输出改进"""
        if "置信度偏低" in feedback:
            return output.replace("置信度: 0.75", "置信度: 0.85") + "\n[改进] 增强实体识别模型"
        if "过于简单" in feedback:
            return output + "\n[改进] 添加属性分析: 静音效果、吸力强度、舒适度"
        return output + "\n[改进] 优化输出格式"

# ============================================================
# 4. 反馈闭环编排器 (Feedback Loop Orchestrator)
# ============================================================

class FeedbackLoopOrchestrator:
    """反馈闭环编排器：整合记忆库和自我反思引擎"""
    def __init__(self, memory_bank: MemoryBank, refinement_engine: SelfRefinementEngine):
        self.memory_bank = memory_bank
        self.refinement_engine = refinement_engine
        self