---
name: paper-workflow
description: This skill should be used when the user asks to "运行paper2skills流程", "生成skill卡片", "处理学术论文", "将论文转为技能", "paper转skill", "论文萃取". Provides the complete workflow for converting academic papers into business-ready skill cards for cross-border e-commerce applications.
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

- 根据业务需求选择领域（因果推断、A/B实验、时间序列、供应链、推荐系统、增长模型）
- 使用关键词库搜索相关论文
- 评估论文的实用性和可落地性

### Step 2: 论文萃取

使用 `paper-萃取` skill 生成 Skill 卡片：

- 读取论文内容和摘要
- 应用 Master Prompt 生成完整 Skill 卡片
- 生成可运行的 Python 代码模板

### Step 3: 质量审核

使用 `paper-审核` skill 进行质量检查：

- 检查算法原理是否用自己的话重述
- 检查应用案例是否具体明确
- 检查代码是否完整可运行
- 检查商业价值评估是否有量化依据

### Step 4: 多端同步

使用 `paper-同步` skill 同步到各平台：

- 同步到 Obsidian vault
- 同步到 GitHub 代码仓库
- 同步到飞书/Notion（可选）

## 触发方式

当用户提及以下内容时触发此 skill：

- "运行 paper2skills 流程"
- "生成 skill 卡片"
- "处理学术论文"
- "将论文转为技能"
- "paper 转 skill"
- "论文萃取"

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
- **`../paper2skills-vault/07-资源库/MasterPrompt.md`**

### 关键词库

论文搜索关键词库：
- **`../paper2skills-vault/07-资源库/关键词库.md`**

### 已有 Skill 卡片

已生成的 Skill 卡片：
- **`../paper2skills-vault/01-因果推断/Skill-Uplift-Modeling.md`**

### 代码模板

Python 代码模板：
- **`../paper2skills-code/`**

## 子技能详情

### paper-选题

负责从 ArXiv 筛选论文。

### paper-萃取

负责使用 Master Prompt 生成 Skill 卡片。

### paper-审核

负责质量检查和修改。

### paper-同步

负责多端同步。
