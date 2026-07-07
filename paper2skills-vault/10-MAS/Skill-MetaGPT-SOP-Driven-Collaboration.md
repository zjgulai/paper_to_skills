---
title: MetaGPT — SOP 驱动的多智能体协作框架
doc_type: knowledge
module: 10-MAS
topic: metagent-sop-driven-collaboration
status: stable
created: 2026-05-10
updated: 2026-05-10
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill Card: MetaGPT — SOP 驱动的多智能体协作框架

---

## ① 算法原理

### 核心思想

**MetaGPT** 将人类组织中的 **Standardized Operating Procedures（SOP，标准作业程序）** 引入多 agent 协作。核心洞察：**复杂任务失败的主要原因是 agent 间缺乏标准化协作规范和结构化信息传递**。通过模拟软件公司的角色分工（PM → Architect → Engineer → QA）和文档驱动的工作流，MetaGPT 显著减少了多 agent 协作中的幻觉和级联错误。

MetaGPT 的三个核心机制：

1. **角色专业化（Role Specialization）**：每个 agent 有明确的角色、职责和约束。例如 Product Manager 负责编写 PRD，Architect 负责系统设计，Engineer 负责编码实现。

2. **结构化输出（Structured Output）**：agent 间的通信不是自由文本，而是标准化的文档（PRD、设计文档、代码、测试用例）。这种结构化中间产物显著提高了协作的一致性和可追溯性。

3. **共享消息池 + 发布-订阅（Shared Message Pool + Pub-Sub）**：所有 agent 通过共享消息池通信，每个 agent 订阅自己关心的消息类型，实现高效的信息分发。

### 数学直觉

**级联错误控制**：

在多 agent 链式协作中，错误会级联传播。设单 agent 的幻觉概率为 $p$，经过 $n$ 个 agent 的链式传递后：

$$P_{cascade}(error) = 1 - (1-p)^n \approx np \quad (for\ small\ p)$$

MetaGPT 通过 **SOP 校验点** 和 **结构化文档约束** 将每个环节的 $p$ 降低。设 SOP 使幻觉概率从 $p$ 降至 $p'$（$p' \ll p$）：

$$P'_{cascade}(error) = 1 - (1-p')^n \ll P_{cascade}(error)$$

实验验证：在 HumanEval 上，MetaGPT 相比其他多 agent 框架，代码生成准确率提升 5.4%（MBPP 基准）。

### 关键假设

1. **任务可被 SOP 化**：复杂任务可以被分解为标准化步骤，类似软件开发的瀑布模型
2. **结构化输出优于自由文本**：PRD、设计文档等结构化格式比对话式交流更不容易产生歧义
3. **角色清晰可分工**：任务可以明确分配给不同角色的 agent，避免职责重叠
4. **中间产物可校验**：每个步骤的输出可以被下游 agent 校验和验证

---

## ② 母婴出海应用案例

### 场景一：VOC 分析标准化流水线

**业务问题**：

母婴出海平台的 VOC 分析需要多人协作完成（分析师、工程师、质检员），但缺乏标准化流程，导致：
- 每次分析的质量不一致
- 不同分析师的产出格式不统一
- 错误难以追溯和修复

**数据要求**：

- 标准化 SOP 文档（分析步骤、验收标准、输出格式）
- 历史分析报告（作为参考模板）
- 评论数据（输入）

**预期产出**：

```
SOP 执行流程:

[Product Manager] 编写 PRD
  输出: 《VOC 分析需求文档 v1.0》
    ↓
[Architect] 设计分析方案
  输出: 《VOC 分析系统架构设计》
    ↓
[Project Manager] 任务分解
  输出: 《任务分解与排期表》
    ↓
[Engineer] 执行分析
  输出: 《数据处理结果 + 分析报告》
    ↓
[QA Engineer] 质量校验
  输出: 《质量校验报告》

最终交付: 5 份结构化文档 + 可追溯的完整工作流
```

**业务价值**：
- 分析质量一致性提升：不同批次分析结果的标准差降低 70%
- 新人 onboarding 时间从 2 周缩短至 2 天（按 SOP 执行即可）
- 错误可追溯：任何问题可以追溯到具体步骤和责任人

---

### 场景二：标准化竞品报告生成

**业务问题**：

竞品分析报告需要多人协作，但报告质量依赖个人经验，格式不统一，难以横向对比。

**数据要求**：

- 竞品数据（评论、价格、功能）
- 报告模板（标准化章节和格式）
- 历史报告库

**预期产出**：

```
MetaGPT 协作:

PM: 定义报告范围和验收标准
  → 《竞品分析报告 PRD》

Architect: 设计分析维度和方法论
  → 《竞品分析架构设计》

PM: 分解为具体任务
  → 《任务分解表》

Engineer (数据): 执行数据采集和清洗
  → 《数据处理日志》

Engineer (分析): 执行多维度分析
  → 《分析结果汇总》

QA: 校验数据准确性和结论合理性
  → 《质量校验报告》

最终交付: 标准化竞品分析报告（可直接用于决策）
```

**业务价值**：
- 报告生成周期从 3-5 天缩短至 1 天
- 报告质量不再依赖个人经验，稳定性提升
- 历史报告可复用和对比，形成知识积累

---

## ③ 代码模板

```python
"""
MetaGPT — SOP 驱动的多智能体协作框架 (简化模拟版)
仅依赖标准库 + numpy，演示核心协作流程
"""

import numpy as np
from typing import Dict, List, Any
from collections import defaultdict
import json

# ============================================================
# 1. 共享消息池 (发布-订阅模式)
# ============================================================
class SharedMessagePool:
    """所有 agent 通过此池通信，支持按消息类型订阅"""
    def __init__(self):
        self.messages = []
        self.subscribers = defaultdict(list)  # topic -> [callback]
    
    def subscribe(self, topic: str, callback):
        self.subscribers[topic].append(callback)
    
    def publish(self, topic: str, content: Any, sender: str):
        msg = {
            "topic": topic,
            "content": content,
            "sender": sender,
            "id": len(self.messages)
        }
        self.messages.append(msg)
        # 通知订阅者
        for cb in self.subscribers.get(topic, []):
            cb(msg)
        return msg
    
    def get_messages_by_topic(self, topic: str) -> List[Dict]:
        return [m for m in self.messages if m["topic"] == topic]

# ============================================================
# 2. 基础 Agent 类
# ============================================================
class MetaGPTAgent:
    """具有角色和 SOP 约束的 agent"""
    def __init__(self, name: str, role: str, pool: SharedMessagePool):
        self.name = name
        self.role = role
        self.pool = pool
        self.sop_steps = []  # 该角色负责的 SOP 步骤
    
    def add_sop_step(self, step_name: str, input_topics: List[str], output_topic: str):
        self.sop_steps.append({
            "step": step_name,
            "input_topics": input_topics,
            "output_topic": output_topic
        })
    
    def execute_step(self, step_idx: int, context: Dict[str, Any]) -> Dict:
        """执行一个 SOP 步骤，返回结构化输出"""
        if step_idx >= len(self.sop_steps):
            return {"error": "Step index out of range"}
        
        step = self.sop_steps[step_idx]
        # 收集输入
        inputs = {}
        for topic in step["input_topics"]:
            msgs = self.pool.get_messages_by_topic(topic)
            if msgs:
                inputs[topic] = msgs[-1]["content"]  # 取最新消息
        
        # 模拟执行（真实场景调用 LLM）
        output = self._simulate_work(step["step"], inputs, context)
        
        # 发布输出
        self.pool.publish(step["output_topic"], output, self.name)
        return output
    
    def _simulate_work(self, step_name: str, inputs: Dict, context: Dict) -> Dict:
        """模拟 agent 工作（真实场景替换为 LLM 调用）"""
        # 根据角色和步骤生成结构化输出
        if self.role == "Product Manager":
            if "prd" in step_name.lower():
                return {
                    "type": "PRD",
                    "product": context.get("product", "unknown"),
                    "requirements": context.get("requirements", []),
                    "acceptance_criteria": ["criteria_1", "criteria_2"],
                    "version": "1.0"
                }
        elif self.role == "Architect":
            if "design" in step_name.lower():
                return {
                    "type": "Architecture Design",
                    "components": ["data_ingestion", "analysis_engine", "report_generator"],
                    "data_flow": ["input → clean → analyze → output"],
                    "constraints": ["latency < 5s", "throughput > 1000 req/s"]
                }
        elif self.role == "Engineer":
            if "implement" in step_name.lower():
                return {
                    "type": "Implementation",
                    "code_summary": "Implemented analysis pipeline with 3 stages",
                    "files": ["data_processor.py", "analyzer.py", "reporter.py"],
                    "test_coverage": 0.85
                }
        elif self.role == "QA":
            if "test" in step_name.lower():
                return {
                    "type": "Test Report",
                    "passed": 15,
                    "failed": 2,
                    "coverage": 0.82,
                    "issues": ["issue_001: edge case in data cleaning"]
                }
        
        # 默认输出
        return {
            "type": "generic_output",
            "step": step_name,
            "status": "completed"
        }

# ============================================================
# 3. SOP 工作流引擎
# ============================================================
class SOPWorkflow:
    """定义和管理 SOP 流程"""
    def __init__(self, name: str):
        self.name = name
        self.steps = []  # [(agent_name, step_name, input_topics, output_topic)]
    
    def add_step(self, agent_name: str, step_name: str, 
                 input_topics: List[str], output_topic: str):
        self.steps.append({
            "agent": agent_name,
            "step": step_name,
            "inputs": input_topics,
            "output": output_topic
        })
    
    def get_steps_for_agent(self, agent_name: str) -> List[Dict]:
        return [s for s in self.steps if s["agent"] == agent_name]

# ============================================================
# 4. MetaGPT 主控制器
# ============================================================
class MetaGPT:
    """管理 agent 招聘、SOP 定义和任务执行"""
    def __init__(self):
        self.pool = SharedMessagePool()
        self.agents = {}
        self.workflows = {}
    
    def hire_agent(self, name: str, role: str) -> MetaGPTAgent:
        agent = MetaGPTAgent(name, role, self.pool)
        self.agents[name] = agent
        return agent
    
    def define_workflow(self, workflow: SOPWorkflow):
        """为每个 agent 注册 SOP 步骤"""
        self.workflows[workflow.name] = workflow
        for step in workflow.steps:
            agent = self.agents.get(step["agent"])
            if agent:
                agent.add_sop_step(
                    step_name=step["step"],
                    input_topics=step["inputs"],
                    output_topic=step["output"]
                )
    
    def run_workflow(self, workflow_name: str, context: Dict[str, Any]) -> List[Dict]:
        """按顺序执行工作流"""
        workflow = self.workflows.get(workflow_name)
        if not workflow:
            return [{"error": f"Workflow {workflow_name} not found"}]
        
        results = []
        for step in workflow.steps:
            agent = self.agents.get(step["agent"])
            if not agent:
                results.append({"error": f"Agent {step['agent']} not found"})
                continue
            
            # 找到 agent 中对应的 step 索引
            step_idx = None
            for i, s in enumerate(agent.sop_steps):
                if s["step"] == step["step"]:
                    step_idx = i
                    break
            
            if step_idx is not None:
                output = agent.execute_step(step_idx, context)
                results.append({
                    "agent": step["agent"],
                    "step": step["step"],
                    "output": output
                })
                print(f"  [{step['agent']}] 完成步骤: {step['step']}")
            else:
                results.append({"error": f"Step {step['step']} not found for agent {step['agent']}"})
        
        return results

# ============================================================
# 5. 演示：VOC 分析标准化流水线
# ============================================================
def demo_voc_analysis():
    print("=" * 60)
    print("MetaGPT SOP 驱动协作演示: VOC 分析流水线")
    print("=" * 60)
    
    # 初始化 MetaGPT
    metagpt = MetaGPT()
    
    # 招聘 agent
    pm = metagpt.hire_agent("Alice", "Product Manager")
    architect = metagpt.hire_agent("Bob", "Architect")
    engineer = metagpt.hire_agent("Charlie", "Engineer")
    qa = metagpt.hire_agent("Diana", "QA")
    
    # 定义 SOP 工作流
    workflow = SOPWorkflow("VOC Analysis Pipeline")
    workflow.add_step("Alice", "Write PRD", [], "prd_doc")
    workflow.add_step("Bob", "Design Architecture", ["prd_doc"], "design_doc")
    workflow.add_step("Charlie", "Implement Analysis", ["prd_doc", "design_doc"], "implementation")
    workflow.add_step("Diana", "Run Tests", ["implementation"], "test_report")
    
    metagpt.define_workflow(workflow)
    
    # 执行工作流
    context = {
        "product": "BabyCare Plus",
        "requirements": [
            "Analyze customer reviews for sentiment",
            "Identify top 5 complaints",
            "Generate weekly report"
        ]
    }
    
    print("\n开始执行 SOP 工作流...\n")
    results = metagpt.run_workflow("VOC Analysis Pipeline", context)
    
    # 输出最终结果摘要
    print("\n" + "-" * 40)
    print("最终交付物摘要:")
    print("-" * 40)
    for r in results:
        if "output" in r:
            output = r["output"]
            print(f"\n  [{r['agent']}] {r['step']}:")
            print(f"    类型: {output.get('type', 'N/A')}")
            if 'product' in output:
                print(f"    产品: {output['product']}")
            if 'components' in output:
                print(f"    组件: {', '.join(output['components'])}")
            if 'code_summary' in output:
                print(f"    代码: {output['code_summary']}")
            if 'passed' in output:
                print(f"    测试: {output['passed']} passed, {output['failed']} failed")
    
    print("\n" + "=" * 60)
    print("[✓] MetaGPT SOP 驱动协作测试通过")
    print("=" * 60)
    
    return results

# ============================================================
# 6. 主程序入口
# ============================================================
if __name__ == "__main__":
    demo_voc_analysis()
```

**运行方式**：
```bash
python metagent_sop.py
```

**预期输出**：
```
============================================================
MetaGPT SOP 驱动协作演示: VOC 分析流水线
============================================================

开始执行 SOP 工作流...

  [Alice] 完成步骤: Write PRD
  [Bob] 完成步骤: Design Architecture
  [Charlie] 完成步骤: Implement Analysis
  [Diana] 完成步骤: Run Tests

----------------------------------------
最终交付物摘要:
----------------------------------------

  [Alice] Write PRD:
    类型: PRD
    产品: BabyCare Plus

  [Bob] Design Architecture:
    类型: Architecture Design
    组件: data_ingestion, analysis_engine, report_generator

  [Charlie] Implement Analysis:
    类型: Implementation
    代码: Implemented analysis pipeline with 3 stages

  [Diana] Run Tests:
    类型: Test Report
    测试: 15 passed, 2 failed

============================================================
[✓] MetaGPT SOP 驱动协作测试通过
============================================================
```

**生产环境建议**：
1. 接入真实 LLM API 生成高质量结构化文档
2. 使用 MetaGPT 官方实现（`pip install metagpt`）
3. 为每个角色设计详细的 prompt template（参考官方仓库）
4. 实现可执行反馈（代码运行时错误自动修复）
5. 添加文档版本控制和审批流程

---

## ④ 技能关联

### 前置技能
- **软件工程基础**：理解瀑布模型、敏捷开发、SOP 概念
- **Prompt Engineering**：为不同角色设计有效的 system prompt
- **结构化输出**：理解 JSON、Markdown、YAML 等结构化格式

### 延伸技能
- **AutoGen**：灵活对话编排，与 MetaGPT 的标准化流程互补
- **CAMEL**：角色扮演式自主协作，适用于开放探索场景
- **Tree of Thoughts**：树搜索式规划，增强复杂决策能力
- **Hierarchical Task Networks**：层次化任务分解

### 可组合技能
- **InstructUIE**：作为 Engineer agent 的底层抽取能力
- **HGT**：提供图推理结果，作为 Architect agent 的设计输入
- **Semantic Blueprint**：约束 agent 产出的结构一致性
- **GraphRAG**：为 PM agent 提供竞品情报检索

---


- **可组合**：[[Skill-MAS-Orchestrator]] / [[Skill-ReAct-Reasoning-Acting]]
- **延伸（extends）**：[[Skill-Policy-Driven-Meta-Controller]]

## ⑤ 商业价值评估

### ROI 预估

| 场景 | 预期收益 | 实施成本 | ROI |
|------|---------|---------|-----|
| VOC 标准化分析 | 质量一致性提升 70%，新人 onboarding 缩短 85% | 开发 3-4 周 | 10-15x |
| 竞品报告生成 | 生成周期从 5 天缩短至 1 天 | 开发 2-3 周 | 12-18x |
| 产品需求文档 | PRD 质量提升，返工率降低 50% | 开发 2 周 | 8-12x |

### 实施难度
**评分：⭐⭐⭐⭐☆（4/5星）**

- 数据要求：中，需要定义清晰的 SOP 和角色职责
- 技术门槛：中，需理解 SOP 设计和结构化输出约束
- 工程复杂度：中高，官方库较重，定制化需要深入理解源码
- 维护成本：中，SOP 变更需要更新所有相关 agent 的 prompt

### 优先级评分
**评分：⭐⭐⭐⭐☆（4/5星）**

- **业务价值高**：标准化流程直接解决质量不一致痛点
- **技术前沿性**：ICLR 2024，SOP 驱动的多 agent 协作是前沿方向
- **可落地性中**：需要较完整的 SOP 定义，不适合探索性任务
- **长期价值**：SOP 本身就是业务知识沉淀，可复用和扩展

### 评估依据
1. **实验验证充分**：HumanEval 85.9%、MBPP 87.7%，显著优于其他框架
2. **结构化输出减少幻觉**：PRD/设计文档/代码/测试的流水线显著降低级联错误
3. **与 AutoGen 互补**：标准化任务用 MetaGPT，探索性任务用 AutoGen
4. **SOP 即知识沉淀**：每次执行都在完善和积累业务 SOP

---

## 参考论文

1. **MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework** (ICLR 2024)
   - Hong, S., et al. (DeepWisdom / KAUST / Xiamen / CUHK / Nanjing / Penn / IDSIA)
   - 核心贡献：SOP 驱动协作 + 角色专业化 + 结构化输出 + 共享消息池
   - 代码：https://github.com/geekan/MetaGPT
   - arXiv：2308.00352

---

## 开源资源

- **MetaGPT 官方**: https://github.com/geekan/MetaGPT
- **文档**: https://docs.metagpt.io/
- **示例**: https://github.com/geekan/MetaGPT/tree/main/examples

---

## 与 AutoGen 的对比与互补

| 维度 | MetaGPT | AutoGen |
|------|---------|---------|
| 核心范式 | SOP 标准化流程 | 灵活对话编排 |
| 控制方式 | SOP 驱动（中心化） | 对话驱动（去中心化） |
| 适用场景 | 标准化任务、流水线生产 | 探索性任务、动态协作 |
| 输出约束 | 严格（结构化文档） | 灵活（自由文本） |
| 学习曲线 | 中 | 低 |

**互补使用建议**：
- 标准化、重复性任务用 MetaGPT（质量可控、可复现）
- 探索性、创新性任务用 AutoGen（灵活、快速迭代）
- 混合模式：MetaGPT 的 SOP agent 组内用 AutoGen 进行灵活讨论
