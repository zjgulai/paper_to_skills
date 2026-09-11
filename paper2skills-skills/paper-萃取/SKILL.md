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
- "生成 skill 卡片" 或 "生成技能卡片"
- "萃取论文" 或 "论文萃取"
- "论文转 skill" 或 "论文转技能"
- "从论文创建 skill"

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
2. 保存到 `<REPO_ROOT>/paper2skills-vault/papers/[领域]/[论文ID]/paper.pdf`
3. 如果 PDF 不存在，跳过此步骤继续

### Step 3: 应用 Master Prompt

使用 `<REPO_ROOT>/paper2skills-vault/07-资源库/MasterPrompt.md` 中的 Master Prompt 生成 Skill 卡片。

### Step 4: 生成代码模板

根据论文内容，生成可运行的 Python 代码：

- 数据处理
- 核心算法实现
- 示例数据和测试用例

### Step 5: 代码一致性验证（关键步骤 - 强制执行）

**每次生成 Skill 后必须运行 K1 门禁，这是强制流程。未通过验证的 Skill 不得进行 Step 6 保存操作。**

> **2026-09-12 升级**：验证从「人工描述 + `pytest model.py`」改为**调用统一脚本**。
> 原因：`AutoReproduce`(arXiv:2505.20662) 实测，LLM 评审认为"很好"的生成代码
> **执行率仅 17.94%**；`ResearchCodeBench` 附录 G 显示新代码失败中 **58.6% 是语义错误**
> （能跑但算错）。人工判断无法替代执行凭证。

#### 5.1 运行 K1 门禁（一条命令）

```bash
python3 <REPO_ROOT>/paper2skills-skills/paper-萃取/scripts/verify_skill_code.py \
  --card <REPO_ROOT>/paper2skills-vault/<领域>/<Skill-卡片>.md
```

脚本自动完成五级验证：`L1 ast.parse` → `L2 py_compile` → `L3 import 探针`
→ `L4 作为脚本执行` → `L5 pytest 断言`。

#### 5.2 按判定处置（四类，不可混淆）

| 判定 | 含义 | 处置 |
|------|------|------|
| ✅ `PASS` | 真正跑通（L4 执行成功 或 L5 断言全绿） | 可进入 Step 6 |
| 🟡 `ENV_BLOCKED` | 本机缺第三方依赖/凭证（**缺依赖 ≠ 代码正确**） | 可保存但**必须在卡片中标注"未验证"**并补 `requirements`；不得声称已验证 |
| 🔴 `ORPHAN_DEP` | 卡片 `import` 了**仓库内不存在的本地模块** | **必须修复**（补实现或改 import），禁止保存 |
| ❌ `FAIL` | 语法/编译/导入/运行期真实错误 | **必须修复**，禁止保存；错误写入 `07-资源库/审核问题库.md` |

#### 5.3 两条必须遵守的使用约束（实测踩出来的坑）

1. **默认按卡片拼接验证，不要用 `--per-block`** —— 卡片是「block1 定义类 → block3 使用」的
   递进结构，逐块独立导入必然 `NameError`（首轮实测 44 个 L3 失败里 **19 个是假阳性**）。
   `--per-block` 仅用于定位报错块号。
2. **脚本已在断网语义下运行** —— 内置 socket 拦截 + `HF_HUB_OFFLINE=1`，
   使 `from_pretrained()` / `requests.get()` **立即失败并归因为 `ENV_BLOCKED`**，
   而不是挂在网络重试上（首轮实测：挂 30 分钟而 CPU 时间仅 2.2 秒）。
   若卡片确实需要联网或外部数据，请在卡片中显式声明为 `ENV_BLOCKED` 情形。

#### 5.4 归档验证报告（不要手写结论）

```bash
python3 .../verify_skill_code.py --card <卡片> \
  --json-out <REPO_ROOT>/paper2skills-research/data/verification/k1_<卡片名>.json
```

---

### Step 5b: 生成 evidence.md（G2 事实溯源凭证，必做）

对照 `MasterPrompt-v2.md` 的 R1–R5 规则，把卡片中每个「高价值断言」数字的
**逐字原文出处**写入同目录 `evidence.md`。这是通过 K2/G2 门禁的唯一途径。

**背景**：全库实测 **13,868 个数字 vs 仅 3 行**原文引用块，G2 通过率仅 43.8%。
这是硬门禁，不是锦上添花。

```markdown
# evidence.md — <Skill 卡片名>

| # | 卡片中的数字 | 原文逐字摘录 | 出处 |
|---|-------------|-------------|------|
| 1 | 蚕食率 −15pp | "…cannibalization rate dropped by 15 percentage points…" | arXiv:2606.26690 §4.2 (p.7) |
| 2 | +7.20% | "…yielding a 7.20% improvement in…" | arXiv:2608.10182 Table 3 |
```

### Step 5c: 运行 K2 三合一门禁

```bash
python3 <REPO_ROOT>/paper2skills-skills/paper-审核/scripts/gate_check.py \
  --card <REPO_ROOT>/paper2skills-vault/<领域>/<Skill-卡片>.md
```

G1 代码 / G2 事实 / G3 业务**三项全绿**才可进入 Step 6。任一红灯须修复后重跑。

### Step 6: 保存输出（验证通过后）

**前置条件：必须确认 Step 5 代码验证已通过。验证未通过不得执行此步骤。**

保存到相应目录：

- 原始论文PDF: `<REPO_ROOT>/paper2skills-vault/papers/[领域]/[论文ID]/paper.pdf`
- 阅读笔记: `<REPO_ROOT>/paper2skills-vault/papers/[领域]/[论文ID]/notes.md`
- 萃取结果: `<REPO_ROOT>/paper2skills-vault/papers/[领域]/[论文ID]/extract.md`
- Skill 卡片: `<REPO_ROOT>/paper2skills-vault/[领域]/Skill-[算法名称].md`
- 代码模板: `<REPO_ROOT>/paper2skills-code/[领域]/[算法]/model.py`
- 验证报告: `<REPO_ROOT>/paper2skills-vault/papers/[领域]/[论文ID]/verification_report.md`

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
