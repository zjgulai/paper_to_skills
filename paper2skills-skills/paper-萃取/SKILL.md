---
name: paper-萃取
description: This skill should be used when the user asks to "生成skill卡片", "萃取论文", "论文转skill", "生成技能卡片", "创建skill". Uses Master Prompt to convert academic papers into business-ready skill cards.
version: 0.1.0
---

# paper-萃取

使用 Master Prompt 将学术论文转化为 Skill 卡片的技能。

## 概述

读取论文内容，应用 Master Prompt，生成完整的 Skill 卡片和可运行代码。

## 触发方式

用户提及以下内容时触发：
- "生成 skill 卡片"
- "萃取论文"
- "论文转 skill"
- "生成技能卡片"
- "创建 skill"

## 工作流程

### Step 1: 准备论文内容

获取论文的以下信息：

- 标题
- 作者
- arXiv ID
- 摘要
- 核心算法描述
- 关键公式
- PDF原文（下载并保存）

### Step 2: 下载并保存论文PDF

1. 从 arXiv 下载原始 PDF
2. 保存到 `../paper2skills-vault/papers/[领域]/[论文ID]/paper.pdf`
3. 如果 PDF 不存在，跳过此步骤继续

### Step 3: 应用 Master Prompt

使用 `../paper2skills-vault/07-资源库/MasterPrompt.md` 中的 Master Prompt 生成 Skill 卡片。

### Step 4: 生成代码模板

根据论文内容，生成可运行的 Python 代码：

- 数据处理
- 核心算法实现
- 示例数据和测试用例

### Step 5: 代码一致性验证（关键步骤）

**每次生成 Skill 后必须运行代码验证，这是强制流程。**

#### 5.1 提取代码模板

从生成的 Skill 卡片中提取代码部分。

#### 5.2 自动生成测试用例

根据代码模板，生成以下测试用例：

- 基本功能测试
- 边界条件测试
- 典型输入输出测试

#### 5.3 运行代码验证

```bash
cd ../paper2skills-code/[领域]/[算法]/
python -m pytest model.py -v
```

**验证失败处理**：
- 如果代码运行失败，记录错误信息
- 标记 Skill 卡片为"待验证"
- **不允许**将未通过验证的 Skill 保存到 vault
- 需要修复代码后重新验证

#### 5.4 生成验证报告

验证通过后，生成验证报告：

```markdown
## 代码验证报告

- **验证时间**: YYYY-MM-DD HH:MM
- **验证状态**: ✅ 通过 / ❌ 失败
- **测试用例数**: N
- **覆盖率**: XX%
- **执行时间**: XXms

### 测试结果

| 测试用例 | 状态 | 耗时 |
|---------|------|------|
| test_xxx | ✅ | 1ms |
```

### Step 6: 保存输出

保存到相应目录：

- 原始论文PDF: `../paper2skills-vault/papers/[领域]/[论文ID]/paper.pdf`
- 阅读笔记: `../paper2skills-vault/papers/[领域]/[论文ID]/notes.md`
- 萃取结果: `../paper2skills-vault/papers/[领域]/[论文ID]/extract.md`
- Skill 卡片: `../paper2skills-vault/[领域]/Skill-[算法名称].md`
- 代码模板: `../paper2skills-code/[领域]/[算法]/model.py`
- 验证报告: `../paper2skills-vault/papers/[领域]/[论文ID]/verification_report.md`

## Master Prompt 要点

### 角色定义

业务导向的数据科学家，专精于将前沿学术研究成果转化为可落地的商业决策工具。

### 输出格式

1. **算法原理** (≤300字)
   - 核心思想
   - 数学直觉（公式+直观解释）
   - 关键假设

2. **母婴出海应用案例** (1-2个)
   - 业务问题
   - 数据要求
   - 预期产出
   - 业务价值

3. **代码模板**
   - Python
   - 完整可运行
   - 包含测试用例

4. **技能关联**
   - 前置技能
   - 延伸技能
   - 可组合技能

5. **商业价值评估**
   - ROI 预估
   - 实施难度
   - 优先级评分

### 质量要求

- 禁止直接复制论文摘要
- 应用案例必须具体明确
- 代码必须有输入输出定义
- 商业价值必须有量化依据

## 输出结构

### Skill 卡片

```markdown
# Skill Card: [算法名称]

## ① 算法原理
[内容]

## ② 母婴出海应用案例
[场景1]
[场景2]

## ③ 代码模板
```python
[代码]
```

## ④ 技能关联
[内容]

## ⑤ 商业价值评估
[内容]
```

### 代码模板

保存到 `paper2skills-code/[领域]/[算法]/model.py`

## 注意事项

- 确保代码可运行
- 测试用例要完整
- 代码风格要一致
- 遵循项目编码规范
- **代码验证是强制流程**，未通过验证的 Skill 不得保存到 vault
