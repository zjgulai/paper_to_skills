# Paper2Skills 📄→🎯

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active](https://img.shields.io/badge/status-active-brightgreen.svg)]()

> **将前沿学术论文转化为可落地的商业决策技能卡片**
>
> Transform cutting-edge academic papers into actionable business decision skill cards.

---

## 🌟 项目简介 | Project Overview

**Paper2Skills** 是一个专为**母婴出海跨境电商**业务设计的研究转化系统。它将 ArXiv 等学术平台的最新研究成果，通过 AI 辅助萃取和人工审核，转化为可直接应用于业务决策的技能卡片和代码模板。

**核心目标**：
- 📚 每天从学术文献中筛选高价值论文
- 🤖 使用 Master Prompt 自动生成 Skill 卡片
- ✅ 质量审核确保可落地性
- 🔄 多端同步（Obsidian / GitHub / 飞书）

---

## 📊 最新状态 (2026-06-01)

| 维度 | 数值 |
|---|---|
| **Skill 卡片总数** | **263**（22 个领域） |
| **图谱节点数** | **263** |
| **图谱边数** | **3887** |
| **missing_prerequisite 断链** | **0** |
| **HIGH 缺口** | **0** |
| **孤立 Skill** | **0** |
| **MAS 工具注册** | **112 个工具 / 14 个域** |
| **MCP Server** | **4 个 domain server / 28 个工具** |
| **MAS 多智能体系统** | 5 个工作流 / **47 项集成测试全绿** |
| **Python 环境** | Python 3.14 + venv（`.python-version` 锁定） |
| **业务工作流覆盖率** | WF-A 95% / WF-B 90% / WF-C 85% / WF-E 85% / WF-D 80% |
| **累计 ROI** | 12000-22000 万元/年潜在（中型品牌） |

**最近萃取 (2026-05-25, Sprint 4+5 共 12 张)**:

**Sprint 5 — 供应链供应计划侧 (6 张, 2026-05-25)**:
- Promotion-Demand-Decomposition (促销感知需求拆解, SPADE NeurIPS 2024)
- Multi-SKU-Procurement-Budget-Allocation (多SKU采购预算分配, arXiv:2301.02662)
- Dynamic-Lot-Sizing-MOQ (MOQ/价格阶梯动态批量, EJOR 2018+2022)
- Supplier-Capacity-Planning (供应商产能约束排产, arXiv:2402.14506)
- New-Product-Inventory-Coldstart (新品冷启动库存, M&SOM 2019+OR 2023)
- Inventory-Health-Aging-Attribution (库存健康+库龄+供应计划归因, JSCDM 2024+arXiv:2404/2308)

**Sprint 4 — 业务完整性 P0 补缺 (6 张, 2026-05-25)**:
- Listing-Quality-Scoring (Amazon Listing质量评分, KDD 2023 Amazon)
- Product-Lifecycle-Stage (品类生命周期定位, AAAI 2025 PhaseFormer)
- Category-Compliance-Prescan (品类合规风险预筛, ASME IDETC 2025 RECALL-MM) **[新领域 21-合规决策]**
- Market-Size-Estimation (市场容量估算 TAM/SAM, arXiv G-TAB + Bass+GT)
- Data-Drift-Detection (数据漂移检测, arXiv DriftGuard 2026)
- Model-Performance-Monitor (模型性能监控, Champion-Challenger + DriftGuard)

**历史萃取 (2026-05-22, AI视频生成 新领域 8 张)**:
- **P0 商品上架 (2 张)**:AnchorCrafter 虚拟主播带货 / Phantom 商品展示 I2V
- **P1 品牌+UGC (3 张)**:Aquarius 品牌视频 / BrandFusion 品牌植入 / DAWN 口播 Review
- **P2 技术储备 (3 张)**:E-Commerce Benchmark / Text-to-Edit 自动剪辑 / Virbo 多语言 UGC

**历史萃取 (2026-05-21, Sprint 3 完成 12 新 Skill + 18 关联回填)**:
- **12-ML基础 (6 张)** / **v2 P1候选 (6 张)** / **WF-D选品 (4 张)** / **跨域桥梁 (5 张)** / **18-物流 (3 张)** / **19-风控 (3 张)** / **MAS生产化 (3 张)** / **存量精补 (8 张)**

详见:
- [桑基图流量转化Skills整理](~/Desktop/电商流量转化Skills整理.md)
- [Sprint 1+2 迭代总报告](paper2skills-vault/00-项目管理/sprint1-2-iteration-report-20260517.md)
- [6h 迭代总报告](paper2skills-vault/00-项目管理/6h-iteration-report-20260517.md)
- [下一步选题路线图](paper2skills-vault/00-项目管理/next-papers-roadmap.md)

---

## 📁 项目结构 | Project Structure

```
paper_to_skills/
├── 📂 paper2skills-skills/          # Claude Code Skills（工作流技能）
│   ├── paper-workflow/              # 完整工作流编排
│   ├── paper-选题/                   # Step 1: 论文筛选
│   ├── paper-萃取/                   # Step 2: AI萃取生成
│   ├── paper-审核/                   # Step 3: 质量审核
│   └── paper-同步/                   # Step 4: 多端同步
│
├── 📂 paper2skills-vault/           # 知识库（Obsidian 兼容）
│   ├── 01-因果推断/                  # Causal Inference
│   ├── 02-A_B实验/                   # A/B Testing
│   ├── 03-时间序列/                  # Time Series
│   ├── 04-供应链/                    # Supply Chain
│   ├── 05-推荐系统/                  # Recommendation
│   ├── 06-增长模型/                  # Growth Models
│   ├── 08-知识图谱/                  # 知识图谱 / 图神经网络
│   ├── 09-DataAgent-LLM/             # DataAgent & LLM 分析
│   ├── 10-MAS/                       # 多智能体系统
│   ├── 11-AI人文/                     # AI × 人文：疗愈金句、哲理类比
│   ├── 12-ML基础/                     # 机器学习基础
│   ├── 13-广告分析/                    # 广告归因、ROAS 优化
│   ├── 14-用户分析/                    # 漏斗、留存、RFM
│   ├── 15-营销投放分析/                 # MMM、促销效果
│   ├── 16-智能体工程/                    # Agent Skills、Context、MCP/A2A
│   ├── 17-价格优化/                      # 动态定价、竞品监测、清仓捆绑
│   ├── 18-物流履约/                      # 跨境路径、配送时效、退货
│   ├── 19-风控反欺诈/                    # 虚假评论、异常交易、刷量
│   ├── 20-AI视频生成/                    # 虚拟主播、商品展示、UGC口播
│   ├── 07-资源库/                    # MasterPrompt / 关键词库
│   └── papers/                       # 原始论文存档
│
├── 📂 paper2skills-code/            # Python 代码模板
│   ├── causal_inference/            # 因果推断
│   ├── ab_testing/                  # A/B 实验
│   ├── time_series/                 # 时间序列预测
│   ├── supply_chain/                # 供应链优化
│   ├── recommendation/              # 推荐系统
│   ├── growth_model/                # 增长模型
│   ├── nlp_voc/                     # NLP 舆情分析
│   ├── knowledge_graph/             # 知识图谱
│   ├── mas/                         # 多智能体系统
│   ├── advertising/                 # 广告分析
│   ├── user_analytics/              # 用户分析
│   ├── marketing/                   # 营销投放分析
│   ├── pricing/                     # 价格优化
│   ├── logistics/                   # 物流履约
│   ├── risk_fraud/                  # 风控反欺诈
│   └── visual_content/              # AI视频生成
│
├── 📄 CLAUDE.md                     # Claude Code 开发指南
└── 📄 README.md                     # 本文件
```

---

## 🔄 工作流程 | Workflow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   论文选题   │ → │   论文萃取   │ → │   质量审核   │ → │   多端同步   │
│  Selection  │    │ Extraction  │    │   Review    │    │    Sync     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      20%               50%                70%               100%
```

### 详细步骤

| 步骤 | 技能名称 | 功能描述 | 进度 |
|------|---------|---------|------|
| 1 | `paper-选题` | 从 ArXiv/GitHub 筛选高质量论文，排除纯理论/综述类 | 20% |
| 2 | `paper-萃取` | 使用 Master Prompt 生成 Skill 卡片和代码模板 | 50% |
| 3 | `paper-审核` | 5 维度质量评分（算法/案例/代码/关联/价值）| 70% |
| 4 | `paper-同步` | 同步到 Obsidian Vault / GitHub / 飞书 | 100% |

**注意**：Step 2 和 Step 3 可以并行执行，但 Step 4 必须在审核通过后进行。

---

## 🎯 技能领域 | Skill Domains

当前覆盖 **22 大核心领域**，对应母婴出海电商的关键业务场景：

| 领域 | 英文 | 典型应用场景 | 示例 Skill |
|------|------|-------------|-----------|
| 因果推断 | Causal Inference | 广告投放归因、促销效果评估 | Uplift Modeling |
| A/B 实验 | A/B Testing | 转化率测试、定价策略验证 | Multi-Armed Bandit |
| 时间序列 | Time Series | 销量预测、需求预测、库存计划 | Demand Forecasting |
| 供应链 | Supply Chain | 供应计划、备货策略、库存优化 | Promotion-Demand-Decomposition |
| 推荐系统 | Recommendation | 复购推荐、搜索排序、首页推荐 | Matrix Factorization |
| 增长模型 | Growth | 拉新、促活、留存、LTV 预测、市场容量估算 | Churn Prediction / Market-Size-Estimation |
| 知识图谱 | Knowledge Graph | 异构表示学习、层次嵌入、语义检索 | HGT / HGCN |
| DataAgent-LLM | DataAgent | LLM 驱动的数据分析、根因分析 | SQL Agent / RCA |
| 多智能体系统 | MAS | Agent 协作、任务规划、编排执行 | AutoGen / ReAct |
| AI人文 | AI Humanities | AI 概念的人文类比、疗愈金句 | LoRA / Prompt Tuning |
| ML基础 | ML Fundamentals | 特征工程、模型评估、漂移检测、性能监控 | Data-Drift-Detection / Model-Performance-Monitor |
| 广告分析 | Advertising | 多触点归因、ROAS 优化、Listing 质量 | Listing-Quality-Scoring / Ad-Attribution |
| 用户分析 | User Analytics | 漏斗分析、留存分析、用户分群 | Cohort / Funnel / RFM |
| 营销投放分析 | Marketing | MMM、促销效果评估、因果 ML | Meridian / DML |
| 智能体工程 | Agent Engineering | Agent Skills、Context、MCP/A2A | Skill Registry |
| 价格优化 | Pricing | 动态定价、竞品监测、清仓捆绑 | Dynamic Pricing |
| 物流履约 | Logistics | 跨境路径、配送时效、退货 | Cross-Border Routing |
| 风控反欺诈 | Risk & Fraud | 虚假评论、异常交易、刷量 | Review Fraud |
| AI视频生成 | Visual Content | 虚拟主播、商品展示、品牌视频、UGC口播 | AnchorCrafter / Phantom |
| **合规决策** | **Compliance** | **品类合规预筛、监管风险、合规护城河** | **Category-Compliance-Prescan** |

---

## 📋 Skill 卡片格式

每个 Skill 卡片包含 5 个标准化模块：

### ① 算法原理（≤300字）
- **核心思想**：一句话概括算法解决的问题
- **数学直觉**：关键公式 + 直观解释
- **关键假设**：算法有效的使用条件

### ② 母婴出海应用案例（1-2个具体场景）
- 业务问题：具体的业务痛点
- 数据要求：需要什么数据、什么格式
- 预期产出：能产出什么结果
- 业务价值：量化商业价值

### ③ 代码模板（Python）
- 完整可运行的 Python 代码
- 包含示例数据和测试用例
- 业务分析导向的封装设计

### ④ 技能关联
- 前置技能：学习此技能前需要掌握什么
- 延伸技能：学会后可以深入的方向
- 可组合：与哪些 Skill 组合效果更好

### ⑤ 商业价值评估
- ROI 预估：量化预期收益
- 实施难度：⭐☆☆☆☆（1-5星）
- 优先级评分：⭐⭐⭐☆☆（1-5星）

---

## 🚀 快速开始 | Quick Start

### 1. 克隆仓库

```bash
git clone https://github.com/zjgulai/paper_to_skills.git
cd paper_to_skills
```

### 2. 安装依赖

> **前置条件（macOS）：** `brew install libomp`（lightgbm/causalml 运行时需要）
>
> **Python 版本：** 3.14+（`cat .python-version` 查看锁定版本）

```bash
# 建议使用 venv 隔离环境（PEP 668，系统 Python 禁止直接 pip install）
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 安装精确锁定版本（推荐，可复现）
pip install -r paper2skills-code/requirements-lock.txt

# 或使用宽松版本约束
pip install -r paper2skills-code/requirements.txt
```

**MAS 系统**（`mas/`）仅依赖 Python 标准库，无需安装上述包即可运行测试：

```bash
python3 -m pytest mas/tests/ -q
```

### 3. 使用 Claude Code 运行工作流

```bash
# 启动 Claude Code
claude

# 在 Claude Code 中使用自然语言触发
"运行 paper2skills 完整流程"
"筛选因果推断领域的论文"
"生成 Skill-Uplift-Modeling"
```

### 4. 手动同步

```bash
# 同步特定 Skill 到各平台
cd paper2skills-skills/paper-同步
python scripts/sync.py --skill Skill-Uplift-Modeling

# 查看同步状态
python scripts/sync.py --status
```

---

## 📊 质量标准 | Quality Standards

Skill 卡片必须通过以下审核标准（总分 ≥ 7/10）：

| 维度 | 权重 | 最低要求 | 检查要点 |
|------|------|---------|---------|
| 算法原理 | 25% | ≥6分 | 非复制重述、数学直觉清晰 |
| 应用案例 | 25% | ≥6分 | 场景具体、与母婴出海强相关 |
| 代码模板 | 25% | ≥7分 | 完整可运行、有测试用例 |
| 技能关联 | 10% | ≥6分 | 关联≥2个已有 Skill |
| 商业价值 | 15% | ≥6分 | ROI 量化、评分有依据 |

---

## 🔑 核心资源 | Key Resources

| 资源 | 路径 | 说明 |
|------|------|------|
| **Master Prompt** | `paper2skills-vault/07-资源库/MasterPrompt.md` | 论文转 Skill 的核心 Prompt |
| **关键词库** | `paper2skills-vault/07-资源库/关键词库.md` | ArXiv 搜索关键词分类 |
| **审核问题库** | `paper2skills-vault/07-资源库/审核问题库.md` | 常见质量问题记录 |
| **同步状态** | `paper2skills-vault/07-资源库/sync_status.json` | ⚠️ 自 Phase 3 起停止手动维护（覆盖率 24%），以 vault 文件系统为准；如需重建快照运行 `rebuild_sync_status.py` |
| **开发指南** | `CLAUDE.md` | 给 Claude Code 的开发指南 |

---

## 🛠️ 技术栈 | Tech Stack

- **Python 3.8+**: 核心算法实现
- **scikit-learn**: 机器学习基础
- **causalml / econml**: 因果推断
- **pandas / numpy**: 数据处理
- **prophet / statsmodels**: 时间序列
- **PyTorch / TensorFlow**: 深度学习（可选）

---

## 📝 论文筛选标准 | Paper Selection Criteria

### ✅ 优先选择
- 包含算法/伪代码的论文
- 有完整实验设计和结果
- 有开源代码或工程实现
- 近 3 年发表的论文

### ❌ 排除类型
- Survey / Review / Overview（综述类）
- 纯理论推导无实验验证
- Meta-analysis（元分析）
- Position Paper（立场声明）

---

## 🤝 贡献指南 | Contributing

1. **Fork** 本仓库
2. 创建新的 Skill 分支：`git checkout -b skill/new-skill-name`
3. 遵循 Skill 卡片格式创建内容
4. 确保代码通过质量审核（≥7分）
5. 提交 Pull Request

---

## 📜 许可证 | License

本项目采用 [MIT License](LICENSE) 开源许可证。

---

## 🙏 致谢 | Acknowledgments

- 所有论文原作者和开源社区
- ArXiv 提供的开放获取论文平台
- Claude Code 提供的 AI 辅助开发能力

---

## 📮 联系方式 | Contact

如有问题或建议，欢迎通过以下方式联系：

- 📧 Email: [your-email@example.com]
- 💬 Issues: [GitHub Issues](https://github.com/zjgulai/paper_to_skills/issues)

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请点个 Star 支持一下！** ⭐

*Made with ❤️ for Cross-Border E-Commerce Data Scientists*

</div>
