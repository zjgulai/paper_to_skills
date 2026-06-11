---
name: playbook-generator
description: paper2skills Playbook 生成器，扫描 Skill 卡片与 Skills Graph，生成带目录、搜索、领域页、工作流页、Skill 详情页、智能体广场（12个本地计算引擎）和智能体报告页的静态 HTML 使用手册。当需要发布或更新项目 Playbook 文档站时使用。
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
- `playbook/agents.html`：智能体广场（12个本地计算引擎，无需外部 API）
- `playbook/agent-report.html`：智能体报告（localStorage 持久化，预置24条种子报告）
- `playbook/domains/*.html`：领域页（22个）
- `playbook/topics/*.html`：主题页
- `playbook/workflows/*.html`：工作流页
- `playbook/skills/*.html`：Skill 详情页（398个），含 🧪 调用案例卡片
- `playbook/graph/overview.html`：图谱关系页
- `playbook/playbooks/*.html`：场景手册页（16本）
- `playbook/chat.html`：AI 知识库对话页
- `playbook/ai-roadmap.html`：CEO 能力路线图
- `playbook/build-report.json`：构建报告

## 关键配置文件

```
scripts/config/
├── skill_ps_override.yaml       # 覆盖 problem_solved 字段（业务导向语言）
├── skill_biz_context_override.yaml  # 覆盖业务上下文（角色/触发场景）
└── skill_handbook_map.yaml      # Skills → 场景手册 + 智能体广场的映射
```

## 智能体广场（agents.html）实现说明

12个智能体全部为**本地计算引擎**，不依赖任何外部 API：

- **数值计算型**（5个）：供应链哨兵 / 动态定价顾问 / P&L透视镜 / 广告归因侦探 / 竞品雷达站
  - 每次输入不同数字 → 实时计算不同结果
- **智能模板型**（7个）：Listing医生 / VOC解码器 / 客服分诊台 / 账号卫士 / 品牌合规 / 选品雷达 / TikTok内容官
  - 分析输入文本关键词 → 有针对性的个性化输出

每次运行后通过 `saveReport()` 自动写入 `localStorage('agentReports')`。

## 智能体报告（agent-report.html）实现说明

- 从 `localStorage('agentReports')` 读取历史运行记录
- 预置 `SEED_REPORTS`（24条，每个Agent各2条不同业务场景）
- 通过 `SEED_VERSION = 'v20260611-r2'` 版本化管理，升级时自动合并不覆盖用户数据
- 支持：展开/收起、单条删除、导出TXT、清空

## 重要注意事项

1. **Python f-string 中的 JS 字符串**：避免在 f-string 的 JS 代码中使用单引号/双引号包裹的字符串包含 `\n`，Python 会将其展开为真实换行，导致 JS 语法错误。改用字符串拼接或 `\\n`。

2. **版本号更新**：修改 `SEED_REPORTS` 内容后必须同步更新 `SEED_VERSION`，否则用户浏览器不会刷新种子数据。

3. **build 后必须用 node --check 验证**：
   ```bash
   # 提取并检查 agents.html 和 agent-report.html 的内联 JS
   node --check <(grep -o '<script>.*</script>' playbook/agent-report.html)
   ```

## 部署

```bash
cd playbook && tar -czf /tmp/pb.tar.gz \
  assets/ domains/ graph/ playbooks/ topics/ workflows/ skills/ \
  agents.html agent-report.html ai-roadmap.html index.html chat.html \
  build-report.json README.md
rsync -avz -e "ssh -i ../ai_video.pem" /tmp/pb.tar.gz ubuntu@101.34.52.232:/tmp/
ssh -i ../ai_video.pem ubuntu@101.34.52.232 \
  "rm -rf /opt/paper2skills/html/* && tar -xzf /tmp/pb.tar.gz -C /opt/paper2skills/html/"
```

