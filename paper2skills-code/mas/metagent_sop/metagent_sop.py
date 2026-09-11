"""
MetaGPT — SOP 驱动的多智能体协作框架
基于论文: Hong et al. "MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework" (ICLR 2024)

核心能力:
1. SOP 标准化作业程序 — 将人类最佳实践编码为 agent 协作流程
2. 角色专业化 — Product Manager / Architect / Engineer / QA 等明确分工
3. 结构化输出 — PRD / 设计文档 / 代码 / 测试用例等中间产物
4. 共享消息池 + 发布-订阅通信 — 高效信息传递

母婴电商场景: SOP 驱动的 VOC 分析标准化协作流程
"""

from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json


class Role(Enum):
    """MetaGPT 中的角色类型"""
    PRODUCT_MANAGER = "product_manager"    # 需求分析与 PRD 编写
    ARCHITECT = "architect"                # 系统设计与方案制定
    PROJECT_MANAGER = "project_manager"    # 任务分配与进度管理
    ENGINEER = "engineer"                  # 执行与代码/数据处理
    QA_ENGINEER = "qa_engineer"          # 质量校验与测试


@dataclass
class Document:
    """结构化文档 — MetaGPT 的核心中间产物"""
    doc_type: str           # PRD / Design / Code / Test / Report
    content: str
    author: str
    version: str = "1.0"
    status: str = "draft"   # draft / review / approved


@dataclass
class Task:
    """任务单元"""
    id: str
    description: str
    role: Role
    dependencies: List[str] = field(default_factory=list)
    input_docs: List[Document] = field(default_factory=list)
    output_doc: Optional[Document] = None
    status: str = "pending"  # pending / in_progress / completed / failed


class SOPWorkflow:
    """
    SOP 工作流定义

    每个 SOP 定义了：
    - 参与角色及其职责
    - 任务顺序和依赖关系
    - 中间产物规范
    """

    def __init__(self, name: str, tasks: List[Task], roles: List[Role]):
        self.name = name
        self.tasks = {t.id: t for t in tasks}
        self.roles = roles
        self.documents: List[Document] = []

    def get_executable_tasks(self) -> List[Task]:
        """获取当前可执行的任务（依赖已满足）"""
        executable = []
        for task in self.tasks.values():
            if task.status != "pending":
                continue
            # 检查依赖是否完成
            deps_satisfied = all(
                self.tasks[dep_id].status == "completed"
                for dep_id in task.dependencies
                if dep_id in self.tasks
            )
            if deps_satisfied:
                executable.append(task)
        return executable

    def complete_task(self, task_id: str, output_doc: Document):
        """标记任务完成并存储产物"""
        if task_id in self.tasks:
            self.tasks[task_id].status = "completed"
            self.tasks[task_id].output_doc = output_doc
            self.documents.append(output_doc)


class MetaGPTAgent:
    """
    MetaGPT Agent — 具有特定角色和 SOP 的协作单元

    与 AutoGen 的 ConversableAgent 不同：
    - MetaGPT Agent 严格遵循 SOP 中的角色定义
    - 输出必须是结构化文档（而非自由文本）
    - 通过共享消息池通信（而非直接对话）
    """

    def __init__(self, name: str, role: Role, system_prompt: str):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.message_pool: List[Dict] = []
        self.documents: List[Document] = []

    def subscribe(self, pool: 'SharedMessagePool'):
        """订阅共享消息池"""
        pool.subscribe(self)

    def execute(self, task: Task) -> Document:
        """执行任务并生成结构化输出"""
        # 简化版：基于角色的规则生成（生产环境调用 LLM）
        if self.role == Role.PRODUCT_MANAGER:
            return self._execute_product_manager(task)
        elif self.role == Role.ARCHITECT:
            return self._execute_architect(task)
        elif self.role == Role.PROJECT_MANAGER:
            return self._execute_project_manager(task)
        elif self.role == Role.ENGINEER:
            return self._execute_engineer(task)
        elif self.role == Role.QA_ENGINEER:
            return self._execute_qa(task)
        else:
            return Document("unknown", f"[{self.name}] 未定义的执行逻辑", self.name)

    def _execute_product_manager(self, task: Task) -> Document:
        """PM: 分析需求，编写 PRD"""
        content = f"""# VOC 分析需求文档 (PRD)

## 1. 需求背景
{task.description}

## 2. 分析目标
- 抽取用户反馈中的实体、关系、情感
- 识别产品质量问题和改进方向
- 监控竞品动态和市场趋势

## 3. 验收标准
- 实体抽取准确率 >= 90%
- 情感分类 F1 >= 0.85
- 异常检测响应时间 <= 5分钟

## 4. 交付物
- 结构化抽取结果 (JSON)
- 周度/月度分析报告
- 实时预警通知
"""
        return Document("PRD", content, self.name, status="approved")

    def _execute_architect(self, task: Task) -> Document:
        """Architect: 设计系统架构"""
        prd = task.input_docs[0].content if task.input_docs else ""
        content = f"""# VOC 分析系统架构设计

## 1. 架构概览
```
[评论数据] → [SRL抽取] → [HGT图构建] → [语义蓝图] → [MAS执行]
                ↓            ↓              ↓            ↓
             实体识别     关系推理      结构化约束    Agent编排
```

## 2. 模块设计
- 抽取层: InstructUIE + BERT-SRL
- 图层: HGT + HGCN
- 蓝图层: Schema-Guided Generation
- 执行层: MetaGPT SOP + AutoGen 编排

## 3. 数据流
1. 原始评论 → 预处理 → 结构化抽取
2. 抽取结果 → 异构图构建 → 图推理
3. 推理结果 → 语义蓝图编译 → Task Blueprint
4. Task Blueprint → Agent 分配 → 执行
"""
        return Document("Design", content, self.name, status="approved")

    def _execute_project_manager(self, task: Task) -> Document:
        """PM: 任务分解与分配"""
        content = f"""# 任务分解与排期

## 1. 任务列表
| 任务ID | 描述 | 负责人 | 依赖 | 工期 |
|--------|------|--------|------|------|
| T1 | 数据预处理与清洗 | Engineer | - | 2天 |
| T2 | 实体抽取模型部署 | Engineer | T1 | 3天 |
| T3 | 情感分析模型部署 | Engineer | T1 | 3天 |
| T4 | 异构图构建 | Engineer | T2,T3 | 2天 |
| T5 | 分析报告生成 | Engineer | T4 | 1天 |
| T6 | 质量校验 | QA | T5 | 1天 |

## 2. 里程碑
- M1 (D5): 抽取模型上线
- M2 (D8): 图构建完成
- M3 (D10): 全链路通跑
- M4 (D12): 验收通过
"""
        return Document("Plan", content, self.name, status="approved")

    def _execute_engineer(self, task: Task) -> Document:
        """Engineer: 执行具体任务"""
        content = f"""# 执行结果报告

## 任务: {task.description}

## 执行详情
- 数据量: 1,245 条评论
- 处理时间: 12.3 秒
- 抽取实体: 3,420 个
- 识别关系: 1,890 条
- 情感标注: 1,245 条

## 关键发现
1. 吸奶器品类负面情感上升 15%（主要反馈噪音问题）
2. 储奶袋品类满意度维持 92%
3. 温奶器加热不均投诉增加 3 倍

## 代码提交
- commit: a1b2c3d
- 测试通过率: 98.5%
"""
        return Document("Code", content, self.name, status="completed")

    def _execute_qa(self, task: Task) -> Document:
        """QA: 质量校验"""
        content = f"""# 质量校验报告

## 校验范围
- 抽取结果: 3,420 个实体
- 情感标注: 1,245 条
- 关系识别: 1,890 条

## 校验结果
| 维度 | 样本数 | 准确率 | 状态 |
|------|--------|--------|------|
| 实体边界 | 200 | 94.5% | PASS |
| 实体类型 | 200 | 91.0% | PASS |
| 情感极性 | 200 | 88.5% | PASS |
| 关系正确性 | 100 | 87.0% | PASS |

## 问题清单
1. 3 个实体边界偏移（已修复）
2. 5 条情感极性误判（已修正）

## 结论
质量校验通过，产出物符合验收标准。
"""
        return Document("Test", content, self.name, status="approved")


class SharedMessagePool:
    """共享消息池 — 发布-订阅通信机制"""

    def __init__(self):
        self.subscribers: List[MetaGPTAgent] = []
        self.messages: List[Dict] = []

    def subscribe(self, agent: MetaGPTAgent):
        self.subscribers.append(agent)

    def publish(self, document: Document, sender: str):
        """发布文档到消息池"""
        msg = {
            "type": document.doc_type,
            "content": document.content[:200] + "...",
            "author": document.author,
            "sender": sender,
        }
        self.messages.append(msg)
        # 通知相关订阅者
        for subscriber in self.subscribers:
            if subscriber.role != Role.PRODUCT_MANAGER or document.doc_type == "PRD":
                subscriber.message_pool.append(msg)


class MetaGPT:
    """MetaGPT 主控制器"""

    def __init__(self):
        self.agents: Dict[Role, MetaGPTAgent] = {}
        self.pool = SharedMessagePool()
        self.workflow: Optional[SOPWorkflow] = None

    def hire(self, agent: MetaGPTAgent):
        """招聘 agent（注册到系统）"""
        self.agents[agent.role] = agent
        agent.subscribe(self.pool)

    def define_sop(self, workflow: SOPWorkflow):
        """定义 SOP 工作流"""
        self.workflow = workflow

    def run(self) -> List[Document]:
        """执行 SOP 工作流"""
        if not self.workflow:
            raise ValueError("请先定义 SOP 工作流")

        completed = 0
        max_iterations = 50

        while completed < len(self.workflow.tasks) and max_iterations > 0:
            executable = self.workflow.get_executable_tasks()
            if not executable:
                break

            for task in executable:
                # 找到对应角色的 agent
                agent = self.agents.get(task.role)
                if not agent:
                    continue

                print(f"  [{task.role.value}] {agent.name} 执行任务: {task.id}")

                # 设置输入文档
                for dep_id in task.dependencies:
                    if dep_id in self.workflow.tasks:
                        dep_doc = self.workflow.tasks[dep_id].output_doc
                        if dep_doc:
                            task.input_docs.append(dep_doc)

                # 执行
                output_doc = agent.execute(task)
                self.workflow.complete_task(task.id, output_doc)
                self.pool.publish(output_doc, agent.name)

                completed += 1

            max_iterations -= 1

        return self.workflow.documents


# ============================================
# 母婴电商 VOC SOP 分析流水线
# ============================================

def create_voc_sop_workflow() -> SOPWorkflow:
    """创建 VOC 分析的 SOP 工作流"""

    tasks = [
        Task(
            id="T1",
            description="分析 VOC 分析需求，编写 PRD",
            role=Role.PRODUCT_MANAGER,
            dependencies=[]
        ),
        Task(
            id="T2",
            description="设计 VOC 分析系统架构",
            role=Role.ARCHITECT,
            dependencies=["T1"]
        ),
        Task(
            id="T3",
            description="制定任务分解与排期计划",
            role=Role.PROJECT_MANAGER,
            dependencies=["T2"]
        ),
        Task(
            id="T4",
            description="执行数据抽取与处理",
            role=Role.ENGINEER,
            dependencies=["T3"]
        ),
        Task(
            id="T5",
            description="质量校验与测试",
            role=Role.QA_ENGINEER,
            dependencies=["T4"]
        ),
    ]

    return SOPWorkflow(
        name="VOC_Analysis_SOP",
        tasks=tasks,
        roles=[Role.PRODUCT_MANAGER, Role.ARCHITECT, Role.PROJECT_MANAGER,
               Role.ENGINEER, Role.QA_ENGINEER]
    )


def demo_metagent_voc_pipeline():
    """演示 MetaGPT SOP 驱动的 VOC 分析流程"""
    print("=" * 70)
    print("MetaGPT — SOP 驱动的 VOC 多 Agent 协作")
    print("=" * 70)

    # 1. 创建 MetaGPT 系统
    metagent = MetaGPT()

    # 2. 招聘 agent
    metagent.hire(MetaGPTAgent("Alice", Role.PRODUCT_MANAGER, "PM"))
    metagent.hire(MetaGPTAgent("Bob", Role.ARCHITECT, "Architect"))
    metagent.hire(MetaGPTAgent("Carol", Role.PROJECT_MANAGER, "PM"))
    metagent.hire(MetaGPTAgent("Dave", Role.ENGINEER, "Engineer"))
    metagent.hire(MetaGPTAgent("Eve", Role.QA_ENGINEER, "QA"))

    # 3. 定义 SOP
    sop = create_voc_sop_workflow()
    metagent.define_sop(sop)

    # 4. 执行
    print("\n[SOP 执行开始]\n")
    documents = metagent.run()

    # 5. 输出结果
    print("\n[产出物汇总]")
    for doc in documents:
        print(f"  [{doc.doc_type}] {doc.author} — {doc.status}")
        print(f"    {doc.content[:100]}...")
        print()

    print("=" * 70)
    print(f"SOP 执行完成，共产出 {len(documents)} 份结构化文档")
    print("=" * 70)


if __name__ == "__main__":
    demo_metagent_voc_pipeline()

    print("\n\n生产环境建议:")
    print("  1. 接入真实 LLM API 生成高质量结构化输出")
    print("  2. 使用 MetaGPT 官方实现获得完整功能")
    print("  3. 为每个角色设计详细的 prompt template")
    print("  4. 添加文档版本控制和审批流程")
    print("  5. 实现可执行反馈（代码运行时错误自动修复）")
