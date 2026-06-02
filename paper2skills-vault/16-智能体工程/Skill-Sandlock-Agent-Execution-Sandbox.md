---
title: Sandlock — 轻量 Agent 沙箱：5ms 启动，HTTP ACL，可逆文件系统
doc_type: knowledge
module: 16-智能体工程
topic: sandlock-agent-execution-sandbox
status: stable
created: 2026-06-01
updated: 2026-06-01
owner: self
source: human+ai
---

# Skill Card: Sandlock — 轻量 Agent 执行沙箱

> **领域**: 16-智能体工程 | **类型**: 安全隔离 | **来源**: arXiv:2605.26298

---

## ① 算法原理

**为什么容器/microVM 不适合短命令 Agent**：Docker 容器启动需 500ms-2s，microVM（Firecracker）需 125ms+，对于每次工具调用仅数十毫秒的 Agent 来说开销过大。Sandlock 通过 Rust 实现，启动延迟 **5ms**，专为短命令高频执行设计，Redis 集成零额外开销。

**静态/动态策略分割**：
- **静态层（Landlock + seccomp-bpf）**：进程启动前编译，约束文件系统访问路径（只读/只写白名单）和系统调用集合。静态策略不可在运行时绕过，构成硬性边界。
- **动态层（seccomp 用户通知）**：对需要审查的系统调用（如 `socket`、`connect`）发送通知到监控进程，监控进程根据实时策略决定 allow/deny，支持运行时动态更新。

**可逆文件系统**：写操作先记录到 journal（Copy-on-Write 快照），执行成功提交，失败则自动回滚至初始状态，确保幂等性。

**HTTP 级别 ACL**：在 TCP 连接层之上拦截 HTTP 请求，支持 Method（GET/POST/DELETE）、Host（域名白名单）、路径前缀（`/api/v1/inventory/*`）三维细粒度控制，防止 Agent 访问未授权 API 端点。

---

## ② 母婴出海应用案例

**场景一：WF-A 补货 Agent 沙箱执行**

补货 Agent 在下达采购 PO 前，需在沙箱中模拟完整执行流程：
- 文件写入策略：只允许写 `/tmp/po_draft/`，写入结果可逆（失败自动回滚）
- HTTP ACL：只允许 `POST /api/v1/purchase-orders` 和 `GET /api/v1/inventory`，禁止访问财务系统
- 沙箱通过后人工二次确认，才真正提交 ERP

**实际价值**：2024 年某 DTC 品牌因 Agent 误触发 `DELETE /orders/all` 导致 48 小时订单数据丢失。Sandlock 沙箱隔离完全规避此类风险。

**场景二：WF-D 选品 Agent Python 代码执行**

选品分析 Agent 运行外部 Python 代码分析竞品数据时：
- 网络访问：HTTP ACL 只允许 `GET` 访问 `api.amazon.com` 和 `sellercentral.amazon.com`
- 文件系统：只读挂载竞品数据，写权限限于 `/tmp/analysis_output/`
- 防止数据外泄：禁止访问任何非 Amazon 域名，阻断潜在的数据回传

---

## ③ 代码模板

```python
# paper2skills-code/llm_agent_engineering/sandlock_sandbox/model.py
# 完整实现见代码目录
from paper2skills_code.llm_agent_engineering.sandlock_sandbox.model import (
    SandboxPolicy, SandlockExecutor, ReversibleEffect, HTTPACLChecker,
    HTTPACLRule, CommandResult
)

# WF-A 补货 Agent 沙箱策略
policy = SandboxPolicy(
    readable_paths=["/tmp/inventory_data"],
    writable_paths=["/tmp/po_draft"],
    allowed_tcp_ports=[443],
    http_acl_rules=[
        HTTPACLRule(method="GET",  host="erp.company.com", path_prefix="/api/v1/inventory"),
        HTTPACLRule(method="POST", host="erp.company.com", path_prefix="/api/v1/purchase-orders"),
    ]
)
executor = SandlockExecutor(policy)
result = executor.execute("python generate_po.py --sku B001 --qty 500")
print(result.verdict, result.rollback_applied)  # ALLOW False
```

---

## ④ 技能关联

- **前置**：[[Skill-Agent-Safety-Guardrails]] | [[Skill-Agent-Fault-Tolerance]]
- **延伸**：[[Skill-Progent-Privilege-Control]] | [[Skill-AgentTrust-Runtime-Safety-Interception]]
- **可组合**：[[Skill-SDOF-State-Constrained-Orchestration]] | [[Skill-Atomix-Transactional-Tool-Calls]] | [[Skill-MCP-A2A-Protocol-Stack]]

---

## ⑤ 商业价值

- **核心收益**：Agent 工具执行完全隔离，防止 PO 误操作（删除/超额下单）和数据外泄，年化风险规避价值 **20-50 万元**
- **性能开销**：5ms 启动 vs 容器 500ms，对 Agent 工具调用吞吐量影响 <1%
- **实施难度**：⭐⭐⭐☆☆（需 Linux 环境，Rust 依赖）
- **优先级**：⭐⭐⭐⭐⭐（**P0 生产阻塞**）
- **参考**：arXiv:2605.26298 | GitHub: github.com/multikernel/sandlock
