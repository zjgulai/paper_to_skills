---
name: paper-同步
description: This skill should be used when the user asks to "同步skill", "多端同步", "同步到飞书", "同步到github", "同步到notion". Synchronizes skill cards and code templates to multiple platforms.
version: 0.1.0
---

# paper-同步

将生成的 Skill 卡片和代码模板同步到多个平台的技能。

## 概述

同步内容包括：
- Skill 卡片（Markdown 格式）
- Python 代码模板
- 相关资源文件

## 触发方式

用户提及以下内容时触发：
- "同步 skill 卡片" 或 "多端同步"
- "同步到 vault" 或 "同步到 GitHub"
- "同步到飞书" 或 "同步到 Notion"
- "查看同步状态"

## 同步目标

### 1. Obsidian Vault

本地存储路径：
```
/Users/pray/project/paper_to_skills/paper2skills-vault/[领域]/Skill-[算法名称].md
```

### 2. GitHub 代码仓库

代码存储路径：
```
/Users/pray/project/paper_to_skills/paper2skills-code/[领域]/[算法]/
├── __init__.py
├── model.py
└── example.py
```

### 3. 飞书/Notion（可选）

需要配置：
- 飞书 Webhook URL（配置到 `/Users/pray/.paper2skills/feishu_webhook`）
- Notion API Key（配置到 `/Users/pray/.paper2skills/notion_api_key`）
- 数据库 ID（配置到 `/Users/pray/.paper2skills/notion_db_id`）

## 同步流程

### 自动同步

对于本地目录（Obsidian + GitHub），使用同步脚本：

```bash
# 基本用法：同步到 vault 和 github
python /Users/pray/project/paper_to_skills/paper2skills-skills/paper-同步/scripts/sync.py --skill Skill-Uplift-Modeling.md --target vault,github

# 查看同步状态
python /Users/pray/project/paper_to_skills/paper2skills-skills/paper-同步/scripts/sync.py --skill Skill-Uplift-Modeling.md --status

# 查看所有同步状态
python /Users/pray/project/paper_to_skills/paper2skills-skills/paper-同步/scripts/sync.py --status
```

### 同步状态追踪

同步后，更新同步状态记录：

```
状态记录路径：`/Users/pray/project/paper_to_skills/paper2skills-vault/07-资源库/sync_status.json`
```

状态记录格式：
```json
{
  "Skill-Uplift-Modeling.md": {
    "vault": { "synced": true, "timestamp": "2026-03-28T10:00:00" },
    "github": { "synced": true, "timestamp": "2026-03-28T10:00:01" },
    "feishu": { "synced": false, "error": "not configured" }
  }
}
```

### 手动同步（飞书/Notion）

对于远程平台，需要：
1. 配置 API 凭证（见上方配置路径）
2. 调用相应 API
3. 验证同步结果
4. 更新同步状态

## 目录结构

### Obsidian Vault 结构

```
paper2skills-vault/
├── 00-项目管理/
│   └── 进度追踪.md
├── 01-因果推断/
│   ├── Skill-Uplift-Modeling.md
│   └── assets/
├── 02-A_B实验/
├── 03-时间序列/
├── 04-供应链/
├── 05-推荐系统/
├── 06-增长模型/
└── 07-资源库/
    ├── MasterPrompt.md
    └── 关键词库.md
```

### GitHub 代码结构

```
paper2skills-code/
├── README.md
├── requirements.txt
├── causal_inference/
│   └── uplift_model/
│       ├── __init__.py
│       ├── model.py
│       └── example.py
├── ab_testing/
├── time_series/
├── supply_chain/
├── recommendation/
└── growth_model/
```

## 注意事项

- 同步前确认文件路径正确
- 代码模板需要测试通过后再同步
- 飞书/Notion 同步需要配置凭证
- 建议使用 git 进行版本控制
