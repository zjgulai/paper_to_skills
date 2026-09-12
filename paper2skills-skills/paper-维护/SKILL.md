---
name: paper-维护
description: This skill should be used when the user asks to "仓库体检", "检查仓库", "repo health", "维护检查", "有没有脏数据", or on a weekly cadence. Runs the mechanical health checks over the paper2skills repo — duplicate cards, frontmatter completeness, hardcoded paths, markdown code-fence structure, registry consistency, gate status, repo hygiene — and reports what needs fixing.
version: 0.1.0
---

# paper-维护

把「人工每隔一阵子偶然发现一类脏数据」变成「一条命令跑完、可回归、可自证有效」的检查。

## 什么时候用

- 每周固定体检一次（`PHASE 5 常态化` 的一部分）
- 大批量改动之后（批量出卡、批量改门禁、迁移目录）
- 门禁数字突然变好或变坏时 —— 先体检，确认不是脏数据造成的假象
- 交接前

## 用法

```bash
# 全量体检（人读报告）；有 CRITICAL 时退出码 1
python3 paper2skills-skills/paper-维护/scripts/repo_health.py

# 机器可读，供 CI / 趋势追踪
python3 paper2skills-skills/paper-维护/scripts/repo_health.py \
  --json-out paper2skills-research/data/health/repo_health.json

# 只跑某一项
python3 paper2skills-skills/paper-维护/scripts/repo_health.py --only C4

# 自检：证明检查真的会报警（**改过本脚本后必跑**）
python3 paper2skills-skills/paper-维护/scripts/repo_health.py --selftest
```

## 检查项

每一项都对应**一次真实事故**，不是想象出来的风险。

| ID | 检查 | 对应事故 |
|----|------|----------|
| **C1** | 重复卡片 | `07-NLP-VOC/` 下曾有 26 组同名卡片各存两份 |
| **C2** | frontmatter 完整性 | 130 张卡只有 73 张有 frontmatter、6 张有 `paper:` 溯源字段 |
| **C3** | 硬编码绝对路径 | 曾有 22 处 `/Users/pray/project/paper_to_skills` 失效路径 |
| **C4** | **markdown 代码围栏结构** | 9 处伪代码标成 ```python、1 张嵌套围栏导致后半段代码整个丢失、3 张围栏未闭合 |
| **C5** | registry ↔ 卡片一致性 | registry 记的 `outputs.skill_card` 与磁盘实际不符 |
| **C6** | 门禁汇总 + 产物时效 | 门禁产物过期后会被当成当前结论引用 |
| **C7** | 仓库卫生 | `__pycache__` / `.pyc` / 空目录混进版本控制 |
| **C8** | v2 段落完整性 | 存量卡多为 v1 五段式；缺 ⑥ 必然过不了 G2 |

### C4 为什么值得单列

代码围栏的结构问题**看起来像代码写错了**：K1 报 `ast` 语法错误时，
第一反应是「卡片里的 Python 有问题」，但本仓库实测的 12 处里，
绝大多数是 markdown 结构问题（伪代码标错、嵌套未加长、围栏未闭合）。

⚠️ **但要准确理解 C4 与 K1 的关系**（本脚本早期把这一点写错过）：
- 围栏**未闭合**时 K1 **不会**失败（它把 EOF 当作隐式闭合），所以 **K1 全绿不代表围栏没问题**；
- 围栏未闭合的真实后果是**渲染**：在 Obsidian/GitHub 里，其后的所有正文都会变成代码块；
- 围栏**嵌套未加长**（内层 ``` 而外层也只有 3 个）才会让代码被截断、后半段**整个丢失**。

C4 的硬判定**只认零缩进的围栏**。缩进 1–3 空格的围栏在列表项里合法
（实测：`Skill-DeepAnalyze` 第 259 行是文档里演示 prompt 格式的缩进示例），
早期把它当硬缺陷导致误报，现降级为提示。**假红灯会让整份体检报告失去可信度** ——
这是本仓库反复踩到的教训。

## 与门禁的分工

| | `paper-维护`（本 skill） | `paper-审核/scripts/gate_check.py`（K2 门禁） |
|---|---|---|
| 对象 | **仓库**（跨卡片的整体状态） | **单张卡片**（内容质量） |
| 问题 | 脏数据、结构缺陷、欠账度量 | 代码能不能跑、数字有没有出处、场景够不够具体 |
| 何时 | 每周 / 批量改动后 | 每次出卡、每次同步前 |
| 拦不拦 | 报 CRITICAL，退出码 1 | `sync.py` 默认拦下红灯卡片 |

`paper-维护` 报的是**需要人来决定的欠账**（比如 130 张卡缺 ⑥ 段要按什么节奏补），
不是「自动修好」。它负责让欠账**可见且可回归**，不负责替人做取舍。

## 已知边界（不要高估它）

- **C2 / C8 的分母包含存量卡**：数字难看是事实，不代表改动引入的回归。
  看趋势要看**增量**（新卡是否 100% 合规），不要只看总量。
- **C3 有白名单**：`paper2skills-code/nlp_voc/` 与 `paper2skills-vault/07-NLP-VOC/`
  里的绝对路径是**有意保留**的历史归档（见 `CLAUDE.md` 的 NLP-VOC 迁出说明），不算违规。
- **C4 的「伪代码标成 python」是启发式**（块内含 ASCII 框图字符且无 `def `），
  可能漏掉用中文画的框图。
- **本脚本自己也会出错**：写这一版的过程中，C3 白名单漏了 vault 侧、C4 把缩进围栏当硬缺陷、
  C6 读错了时间戳位置、`critical` 判定把提示当缺陷 —— 共 5 个 bug，全部是**假红灯**。
  所以：**改过脚本后必须跑 `--selftest`**，且任何一次「体检报红」都要先人工确认是真缺陷。

## 修复动作参考

| 发现 | 动作 |
|---|---|
| C1 重复卡片 | 比对字节与内容，保留唯一版本；漂移的移入 `<域>/_superseded/` |
| C2 缺 frontmatter | 按 `MasterPrompt-v2.md` 的 frontmatter 模板补；缺 `paper_id` 的先查 registry |
| C3 硬编码路径 | 可执行代码改 `Path(__file__).resolve().parents[N]` 或 `PAPER2SKILLS_ROOT`；文档用 `<REPO_ROOT>` 占位；历史归档不改写，只加时效注记 |
| C4 围栏未闭合 | 在代码真实结束处补 ```` ``` ```` |
| C4 嵌套未加长 | 外层改用 4 个反引号 |
| C4 伪代码标错 | 改标 ` ```pseudocode ` / ` ```text ` / ` ```yaml ` |
| C5 registry 不符 | 以磁盘实际为准修 registry（registry 是事实源，须与交付物一致） |
| C6 产物过期 | 重跑 K1 / K2 门禁 |
| C7 卫生 | `__pycache__` / `.pyc` 加入 `.gitignore` 并从纳管中移除 |
| C8 缺 ⑥ | 按 `MasterPrompt-v2.md` 的 v2.1 约定补逐字引文（先抓全文底本） |
