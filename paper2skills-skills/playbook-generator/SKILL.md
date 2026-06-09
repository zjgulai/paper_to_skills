---
name: playbook-generator
description: paper2skills Playbook 生成器，扫描 Skill 卡片与 Skills Graph，生成带目录、搜索、领域页、工作流页和 Skill 详情页的静态 HTML 使用手册。当需要发布或更新项目 Playbook 文档站时使用。
---

# Playbook Generator

将 `paper2skills-vault/` 中的 Skill 卡片、领域注册表、Skills Graph 关系和 MAS 工作流信息生成静态 HTML Playbook。

## 使用

```bash
python3 paper2skills-skills/playbook-generator/scripts/build_playbook.py \
  --root . \
  --vault paper2skills-vault \
  --out playbook
```

## 输出

- `playbook/index.html`：总览 Dashboard
- `playbook/domains/*.html`：领域页
- `playbook/topics/*.html`：主题页
- `playbook/workflows/*.html`：工作流页
- `playbook/skills/*.html`：Skill 详情页
- `playbook/graph/overview.html`：图谱关系页
- `playbook/assets/playbook-data.json`：结构化数据
- `playbook/assets/graph-data.json`：图谱数据
- `playbook/build-report.json`：构建报告
