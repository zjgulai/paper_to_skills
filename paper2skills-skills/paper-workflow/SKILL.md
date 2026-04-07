---
name: paper-workflow
description: This skill should be used when the user asks to "run paper2skills workflow", "run complete paper to skill", "batch process papers". Provides the complete workflow for converting academic papers into business-ready skill cards.
version: 0.1.0
---

# paper2skills Workflow

将学术论文转化为可落地的商业决策技能卡片的完整工作流。

## 概述

本工作流包含四个核心子技能：

1. **paper-选题** - 从 ArXiv 筛选高质量论文
2. **paper-萃取** - 使用 Master Prompt 生成 Skill 卡片
3. **paper-审核** - 质量检查和修改
4. **paper-同步** - 多端同步（飞书、GitHub、Notion）

## 工作流程

### Step 1: 论文选题

使用 `paper-选题` skill 筛选论文：

- 根据业务需求选择领域（因果推断、A/B实验、时间序列、供应链、推荐系统、增长模型、NLP-VOC、NPS）
- 使用关键词库搜索相关论文
- 评估论文的实用性和可落地性

**进度**：20%

### Step 2: 论文萃取

使用 `paper-萃取` skill 生成 Skill 卡片：

- 读取论文内容和摘要
- 应用 Master Prompt 生成完整 Skill 卡片
- 生成可运行的 Python 代码模板
- **强制执行代码验证**

**进度**：50%（与 Step 3 可并行）

### Step 3: 质量审核

使用 `paper-审核` skill 进行质量检查：

- 检查算法原理是否用自己的话重述
- 检查应用案例是否具体明确
- 检查代码是否完整可运行
- 检查商业价值评估是否有量化依据
- 使用量化评分表评分（≥7分通过）

**进度**：70%（与 Step 2 可并行执行）

### Step 4: 多端同步

使用 `paper-同步` skill 同步到各平台：

- 同步到 Obsidian vault
- 同步到 GitHub 代码仓库
- 同步到飞书/Notion（可选）
- 更新同步状态追踪

**进度**：100%

## 并行处理

**可并行的步骤**：
- Step 2（论文萃取）和 Step 3（质量审核）可以并行执行
- 当论文内容准备完成后，即可同时进行萃取和审核

**并行执行条件**：
- 论文内容已获取
- Master Prompt 已准备好

**串行依赖**：
- Step 1（选题）必须先完成
- Step 3 必须等 Step 2 的代码生成完成后才能检查代码质量
- Step 4 必须等 Step 3 审核通过后才能同步

## 异常处理

### 常见异常及处理方案

| 异常场景 | 处理方案 | 回滚操作 |
|---------|---------|---------|
| 论文无法下载 | 跳过 PDF，使用摘要继续 | 无需回滚 |
| 代码验证失败 | 记录错误，标记"待验证" | 回到 Step 2 重新生成代码 |
| 审核不通过 | 返回 Step 2 调整内容 | 记录问题，重新萃取 |
| 同步失败 | 重试3次，记录错误 | 无需回滚 |

### 错误状态标记

- `待验证`：代码验证失败，需要修复
- `待调整`：审核不通过，需要修改
- `同步失败`：重试后仍失败，记录错误

## 进度追踪

每个步骤执行时，应向用户报告进度：

```
[paper2skills 流程执行中]
Step 1: 论文选题 ████████░░ 80%
Step 2: 论文萃取 ░░░░░░░░░░░ 0%
Step 3: 质量审核 ░░░░░░░░░░░ 0%
Step 4: 多端同步 ░░░░░░░░░░░ 0%
```

## 可选: Skills Graph 选题增强

在执行完整流程之前，可以使用 `paper-skills-graph` 进行系统性选题分析：

```
用户: "我想系统性完善技能体系"

→ 先运行 paper-skills-graph 分析
   → 发现知识缺口（如: 缺失 LTV 预测技能）
   → 推荐填补缺口的关键词

→ 然后运行 paper-选题 搜索论文
   → 使用推荐的关键词搜索
   → 筛选合适的论文

→ 继续标准 workflow 流程
```

**与 paper-选题 的区别**:
- `paper-选题`: 基于业务需求/关键词主动搜索
- `paper-skills-graph`: 基于知识缺口被动推荐

## 触发方式

当用户提及以下内容时触发此 skill：

- "运行 paper2skills 完整流程"
- "运行 paper workflow"
- "将论文转为 skill 完整流程"
- "批量处理学术论文"
- "系统性完善技能体系" (包含 Skills Graph 分析)

## 详细流程

### 完整流程示例

1. 用户提供论文或业务需求
2. 使用 paper-选题 筛选合适的论文
3. 使用 paper-萃取 生成 Skill 卡片
4. 使用 paper-审核 检查质量
5. 如有质量问题，返回 step 3 调整
6. 使用 paper-同步 同步到各平台

## 参考资源

### Master Prompt

完整的 Master Prompt 存储在：
- **`/Users/pray/project/paper_to_skills/paper2skills-vault/07-资源库/MasterPrompt.md`**

### 关键词库

论文搜索关键词库：
- **`/Users/pray/project/paper_to_skills/paper2skills-vault/07-资源库/关键词库.md`**

### 已有 Skill 卡片

已生成的 Skill 卡片：
- **`/Users/pray/project/paper_to_skills/paper2skills-vault/01-因果推断/Skill-Uplift-Modeling.md`**

### 代码模板

Python 代码模板：
- **`/Users/pray/project/paper_to_skills/paper2skills-code/`**

## 子技能详情

### paper-选题

负责从 ArXiv 筛选论文。

### paper-萃取

负责使用 Master Prompt 生成 Skill 卡片。

### paper-审核

负责质量检查和修改。

### paper-同步

负责多端同步。
