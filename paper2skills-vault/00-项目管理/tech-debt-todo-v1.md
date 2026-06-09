---
name: tech-debt-todo-v1
description: paper2skills 项目技术债、工程债、文档管理债完整清单。基于2026-06-08三路并行深度审计（代码质量+工程基础设施+文档管理）生成。包含42项债务的优先级、估时、验收标准和责任分类。
---

# paper2skills 技术债务完整 TODO v1.0

> 审计日期：2026-06-08 | 涵盖：技术债16项 + 工程债11项 + 文档债15项 = **42项**
> 
> 已完成的 P0 修复标记 ✅

---

## SCQA 背景

**S（情境）** 项目已有 350 Skills、8 工作流、14 手册、414 HTML 页面、部署在 skills.lute-tlz-dddd.top，功能基本完整。

**C（冲突）** 快速迭代积累了大量技术债：单文件5200行的生成器、生成产物提交到 git 导致仓库膨胀12MB、CI 不跑测试、私钥文件游离在 .gitignore 之外、CLAUDE.md 计数偏差127个 Skill。

**Q（问题）** 哪些债务必须立即清偿，哪些可以计划性处理？

**A（答案）** P0 全部立即修复（今天完成），P1 纳入下一个 Sprint，P2/P3 计划性处理。

---

## 已立即完成的 P0 修复

- ✅ **SEC-01** `*.pem` 加入 `.gitignore`（私钥文件安全风险）
- ✅ **CODE-M1** `SECTION_KEYS` 重复 `"技能关系"` 已去重
- ✅ **CODE-M3** YAML 解析异常从静默 `pass` 改为 `stderr` 警告
- ✅ **CODE-H2** `source_excerpt` 死字段已删除（节省 350 × 1.2KB 计算）
- ✅ **DOC-P0-3** `CLAUDE.md` 域 Skill 计数已更新（旧: 233，实际: 350，+127 drift 修复）

---

## 技术债 (Code Quality)

### P1 — 本 Sprint 内处理

- [ ] **CODE-C1** `build_css()` 838行 → 提取为外部 `style.css` + `<link>` 引用
  - **问题**: 838行 CSS 字符串硬编码在 Python 函数里，无 IDE 支持，每个 HTML 页面内联 41KB CSS
  - **修复**: 构建时 `write_file(out/"assets"/"style.css", build_css())` 已有，但 HTML 仍用 `<link>` 引用——现在是内联的，需改为只生成一次
  - **估时**: 2h
  - **验收**: `style.css` 只存在于 `assets/`，每个 HTML `<head>` 有 `<link rel="stylesheet">`，不再有 `<style>` 标签

- [ ] **CODE-D1** 88个技能 `problem_solved == algorithm_summary` → 修复提取逻辑
  - **问题**: 25%（88/350）的技能在详情页的"解决的问题"和"核心算法逻辑"显示完全相同的文本
  - **根因**: `first_bold_sentence(algo_text, ...)` 找不到业务句时退化为 `first_nonempty_line(algo_text)` 和 `algorithm_summary` 同源
  - **修复**: 构建时打印 `WARN: dup_ps {skill_id}` 让问题可见；长期从 `SKILL_PS_OVERRIDE` 批量覆盖
  - **估时**: 3h
  - **验收**: 构建日志无 `dup_ps` 警告，或 `python3 -c "..."` 统计重复数为 0

- [ ] **CODE-H1** `KNOWN_SKILL_IDS` 全局可变状态 → 改为参数传递
  - **问题**: `global KNOWN_SKILL_IDS` 在 `render_pages()` 内写入，在 `link_list()` 内读取，隐式依赖调用顺序
  - **修复**: 将 `KNOWN_SKILL_IDS` 作为参数传给 `link_list(items, skill_ids, nav)`
  - **估时**: 1h
  - **验收**: 文件中无 `global KNOWN_SKILL_IDS`

### P2 — 下个迭代

- [ ] **CODE-H3** `domain_key`/`topic`/`status` 三个未渲染字段清理
  - **问题**: `domain_key` 只在 tags 后备值使用（域名），`topic`（单数）从不渲染，`status` 从未出现在任何 HTML
  - **修复**: 从 `PlaybookSkill` 删除这三个字段定义和赋值，从 `build_skills()` 删除对应赋值
  - **估时**: 1h

- [ ] **CODE-M4** 300行配置数据（`SKILL_PS_OVERRIDE` + `SKILL_BIZ_CONTEXT_OVERRIDE` + `SKILL_HANDBOOK_MAP`）迁移到 YAML 配置文件
  - **问题**: ~400行业务配置数据夹在逻辑代码之间，每次添加技能都要编辑5200行文件
  - **修复**: 提取到 `paper2skills-skills/playbook-generator/config/` 目录，构建启动时加载
  - **估时**: 4h

- [ ] **CODE-C2** `render_roadmap_page()` 354行 → 拆分子函数
  - 拆分为 `_render_rm_hero()`, `_render_rm_phase()`, `_render_rm_footer()`
  - **估时**: 2h

- [ ] **CODE-O2** `WORKFLOW_RULES` 关键词匹配 vs YAML 定义双轨并存 → 职责边界明确化
  - **问题**: 旧关键词匹配体系和新 YAML 定义体系并存，`skill.workflows` 到底表示什么不清晰
  - **修复**: 文档化两者的职责分工；或合并为单一路径
  - **估时**: 1h

### P3 — 技术雷达

- [ ] **CODE-P2** `/skills/index.html` 311KB → 虚拟滚动或服务端分页
  - 350个技能全部内联导致单页 311KB，首屏渲染慢
  - 方案A：客户端虚拟滚动（IntersectionObserver）；方案B：分域分页

- [ ] **CODE-M2** HTML 路径字符串散落（`BUSINESS_ENTRIES` + roadmap footer + CEO tab）→ 中心路由表
  - 同一个 playbook 路径在3处硬编码，重命名需同步修改3处

- [ ] **CODE-P1** CSS minification（41KB → ~25KB）
  - 在 `build_css()` 结尾加 `re.sub(r'\s+', ' ', css)` 简单压缩
  - **估时**: 30min

---

## 工程债 (Engineering Infrastructure)

### P0（已修复）

- ✅ **ENG-01** `ai_video.pem` 加入 `.gitignore`（私钥泄露风险消除）

### P1 — 本 Sprint 内处理

- [ ] **ENG-02** CI 不运行任何测试 → 加测试步骤
  - **现状**: `.github/workflows/build-playbook.yml` 只 build + commit，无 `pytest`
  - **修复**: 在 `Build Playbook` 前加 `pip install pytest && pytest mas/tests -q --tb=short`
  - **估时**: 2h
  - **验收**: CI log 中有 `pytest` 输出；测试失败时 CI 红色

- [ ] **ENG-03** `feat/voc-deep-analysis-mvp` 领先 main 80个提交 → 合并或 PR
  - **现状**: 当前分支领先 `main` 80个提交，积累合并冲突风险
  - **修复**: 创建 PR 到 main，或 `git merge main` 后 squash
  - **估时**: 1h

- [ ] **ENG-04** 423个构建产物提交到 main → 迁出 .gitignore
  - **现状**: `playbook/` 12MB 全部 tracked，每次构建产生 "chore: rebuild playbook" 噪声提交
  - **修复方案**: 
    1. 将 `playbook/` 加入 `.gitignore`（取消注释已有占位符）
    2. CI 改为 deploy 到 `gh-pages` 分支（`peaceiris/actions-gh-pages`）
    3. 或 `rsync` 直接推到腾讯云服务器（`appleboy/ssh-action`）
  - **注意**: 迁移前需确认服务器 `/opt/paper2skills/html` 已设为 auto-deploy
  - **估时**: 4h
  - **验收**: `git ls-files playbook/ | wc -l` = 0

- [ ] **ENG-05** 无自动化 deploy → CI 加 deploy job
  - **现状**: 生产站点 `skills.lute-tlz-dddd.top` 更新 = 手动操作，无可复现流程
  - **修复**: CI 加 deploy job，使用 `appleboy/ssh-action`，secrets: `DEPLOY_KEY`（ai_video.pem 内容）+ `SERVER_HOST`
  - **目标 workflow**:
    ```yaml
    - name: Deploy to server
      uses: appleboy/ssh-action@master
      with:
        host: ${{ secrets.SERVER_HOST }}
        username: ubuntu
        key: ${{ secrets.DEPLOY_KEY }}
        script: |
          cd /opt/paper2skills/html
          # rsync from artifact or pull from gh-pages
    ```
  - **估时**: 3h

### P2 — 下个迭代

- [ ] **ENG-06** `paper2skills-code` 测试与 CI 完全脱离 → 统一测试路径
  - `pyproject.toml` 的 `testpaths = ["mas/tests"]` 不包含 `paper2skills-code/` 的20+个测试文件
  - **估时**: 半天

- [ ] **ENG-07** playbook-generator 无 `requirements.txt` → 创建依赖声明
  - CI 只 `pip install pyyaml`，实际依赖缺乏声明
  - **修复**: 创建 `paper2skills-skills/playbook-generator/requirements.txt` 声明 `pyyaml>=6.0`
  - **估时**: 1h

- [ ] **ENG-08** 无本地预览命令文档
  - **修复**: CLAUDE.md 的 Workflow Commands 增加 `## Local Preview` 命令块：
    ```bash
    python3 -m http.server 8080 --directory playbook
    # 访问 http://localhost:8080
    ```
  - **估时**: 30min

- [ ] **ENG-09** 3个 stale 远端分支清理
  - `origin/cursor/add-agents-md-e7cb`
  - `origin/cursor/fix-tjap-flaky-revenue-test-de3d`
  - `origin/chore/dept-rename-2026-05-13`
  - **命令**: `git push origin --delete cursor/add-agents-md-e7cb cursor/fix-tjap-flaky-revenue-test-de3d chore/dept-rename-2026-05-13`
  - **估时**: 15min

- [ ] **ENG-10** Python 版本不一致（CI: 3.11 vs 代码要求: 3.14）
  - `requirements.txt` 注释写 `Python >= 3.14`，CI 用 3.11
  - **修复**: CI 升级到 3.11+ 并测试兼容性，或降低 requirements.txt 下限
  - **估时**: 2h

### P3

- [ ] **ENG-11** 无构建 diff/changelog 机制
  - 每次 CI 构建覆盖 `build-report.json`，无 Skill 新增/修改/删除 diff
  - **修复**: 构建脚本比对上次 build-report，写入 PR comment 或 workflow summary
  - **估时**: 4h

---

## 文档管理债 (Documentation)

### P0（已修复）

- ✅ **DOC-P0-3** CLAUDE.md 域 Skill 计数更新（233 → 350 实际值，+127 修复）

### P1 — 本 Sprint 内处理

- [ ] **DOC-P1-1** CLAUDE.md: 补充 Sprint 6/7/8 的新 Skill 记录（2026-06-01 到 06-08）
  - **现状**: "Recent Skills Added" 章节停留在 Sprint 5（2026-05-25）
  - **修复**: 从 `进度追踪.md` 提取 Sprint 6/7/8 新增的 ~90 个 Skill 并更新列表
  - **估时**: 1h

- [ ] **DOC-P1-2** CLAUDE.md: 项目结构树补充 `22-数据采集工程` 域
  - **现状**: Project Structure 中缺少 2026-06-05 新增的域
  - **估时**: 15min

- [ ] **DOC-P1-3** 创建部署运行手册（Deployment Runbook）
  - **现状**: 服务器配置、nginx 结构、Docker 命令、更新流程全在人脑记忆中
  - **目标文件**: `paper2skills-vault/00-项目管理/deployment-runbook.md`
  - **内容**: 服务器地址/端口/路径、更新部署命令、nginx 结构说明、SSL 续签步骤、回滚方法
  - **估时**: 2h

- [ ] **DOC-P1-4** `项目总结.md` Skill 总数更新（320 → 350）
  - **现状**: 总结文档标题指标为 "320 个 Skill"，实际 350
  - **估时**: 15min

- [ ] **DOC-P1-5** `next-papers-roadmap-v2.md` checkbox 状态与实际执行对齐
  - **现状**: Sprint 3/4/5 的很多 checkbox 未勾选但实际已完成
  - **修复**: 全部勾选已完成项，添加 Sprint 6/7/8 执行记录
  - **估时**: 1h

### P2 — 下个迭代

- [ ] **DOC-P2-1** 创建独立 Skill Card Format Spec 文件
  - **现状**: 格式规范埋在 CLAUDE.md 散文中
  - **目标**: `paper2skills-vault/07-资源库/SKILL-FORMAT-SPEC.md`（独立可引用）
  - **估时**: 1h

- [ ] **DOC-P2-2** `build_playbook.py` 架构说明文档
  - **现状**: 5200行生成器无任何架构文档，知识在注释和 commit message 里
  - **目标**: `paper2skills-skills/playbook-generator/ARCHITECTURE.md`（流程图 + 关键函数说明）
  - **估时**: 3h

- [ ] **DOC-P2-3** 域扩展指南（如何新增一个域）
  - **现状**: 步骤散落在 CLAUDE.md 各章节，没有 step-by-step 指引
  - **目标**: `paper2skills-vault/00-项目管理/domain-extension-guide.md`
  - **估时**: 2h

- [ ] **DOC-P2-4** `paper2skills_common` 模块 README
  - **现状**: `doctor.py`、`workflow.py`、`evolution.py` 等关键工具无使用说明
  - **估时**: 2h

### P3

- [ ] **DOC-P3-1** `CHANGELOG.md` 主变更日志
  - 从 git log 提取主要 Sprint 里程碑，建立项目级变更历史
  - **估时**: 2h

- [ ] **DOC-P3-2** 内容风格指南（Content Style Guide）
  - 规范母婴跨境场景描述格式、ROI 表达方式、技术术语翻译标准
  - **估时**: 3h

---

## 优先级速查矩阵

| 优先级 | 数量 | 累计估时 | 最高影响项 |
|--------|------|----------|-----------|
| P0（已完成）| 5项 | ~1h | 私钥安全 + 计数修复 |
| P1 | 10项 | ~18h | CI测试 + 构建产物迁移 + 部署自动化 |
| P2 | 10项 | ~23h | CSS提取 + 配置迁移 + 部署文档 |
| P3 | 7项 | ~15h | 虚拟滚动 + CHANGELOG + 风格指南 |
| **合计** | **32项** | **~57h** | |

> P0 已今日完成（5项），剩余 32项需计划执行

---

## 知识孤岛清单（需提取保存）

以下知识仅存在于 git commit / OpenCode session / 个人记忆，需提取到文档：

1. **git commit only**: `22-数据采集工程` 为什么被创建（commit `9ea290c`）
2. **git commit only**: 为什么 NLP-VOC 被迁出到 `../ai_nlp_voc/`
3. **session log only**: Playbook SCQA 叙事框架决策过程（`logs/session-summary-2026-06-05.md`）
4. **人脑记忆**: 腾讯云服务器 nginx 容器 `ai_video_nginx` 的完整挂载结构
5. **人脑记忆**: `fitness-snapshot.json` 评分方法论
6. **代码注释only**: D3 ego graph 设计决策（为什么用 `window._EGO_DATA` 缓存）

---

## 执行建议（Sprint 计划）

**本周（今天开始）**：
1. ENG-03：feat 分支创建 PR 或合并到 main（避免积累更多）
2. ENG-07：创建 generator `requirements.txt`（30分钟）
3. ENG-08：补充本地预览命令（30分钟）
4. ENG-09：清理 3个 stale 分支（15分钟）
5. DOC-P1-1/2/4：CLAUDE.md 补充 + 项目总结更新（1.5小时）
6. DOC-P1-3：部署运行手册（2小时）

**下个 Sprint**：
- ENG-02+ENG-04+ENG-05 打包处理（CI测试 + 构建产物迁移 + deploy job = 9小时，最高工程价值）
- CODE-C1 CSS 提取（和 ENG-04 一起，避免二次修改）
- CODE-D1 88个重复 problem_solved 批量修复

**后续迭代**：
- CODE-M4 配置数据 YAML 化
- DOC-P2-1/2/3 文档补全
