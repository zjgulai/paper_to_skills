---
title: 20-AI视频生成技能索引
doc_type: index
module: 20-AI视频生成
status: active
created: 2026-05-22
updated: 2026-05-22
owner: self
source: ai
---

# 20-AI视频生成 (Visual Content) 技能索引

## 领域定位

为母婴出海电商的品牌推广、商品上架、电商UGC 提供 AI 视频生成能力。覆盖虚拟主播带货、商品展示 I2V、品牌视频生产、口播 Review 等场景。

## 已落地 Skill

| 优先级 | 技能 | 论文 | 代码 | 场景 | ROI |
|--------|------|------|------|------|-----|
| P0 | [Skill-AnchorCrafter-Virtual-Anchor-Demo](./Skill-AnchorCrafter-Virtual-Anchor-Demo.md) | arXiv:2411.17383 | ✅ GitHub | TikTok虚拟主播带货 | 50-100万 |
| P0 | [Skill-Phantom-Product-Showcase-I2V](./Skill-Phantom-Product-Showcase-I2V.md) | arXiv:2502.11079 | ✅ Apache 2.0 | Amazon主图→动态视频 | 150-250万 |
| P1 | [Skill-Aquarius-Brand-Video-Generation](./Skill-Aquarius-Brand-Video-Generation.md) | arXiv:2505.10584 | 🔄 管线即将开源 | 品牌营销视频 DiT | 80-150万 |
| P1 | [Skill-BrandFusion-Multi-Agent](./Skill-BrandFusion-Multi-Agent.md) | arXiv:2603.02816 | 🔄 待发布 | Logo/品牌一致性 | 50-80万 |
| P1 | [Skill-DAWN-Talking-Head-Review](./Skill-DAWN-Talking-Head-Review.md) | arXiv:2410.13726 | ✅ GitHub | UGC口播Review | 30-60万 |
| P2 | [Skill-E-Commerce-Video-Benchmark](./Skill-E-Commerce-Video-Benchmark.md) | ICLR 2026 | ✅ 数据集 | 电商视频质量评测 | 技术储备 |
| P2 | [Skill-Text-to-Edit-Video-Ad](./Skill-Text-to-Edit-Video-Ad.md) | arXiv:2501.05884 | ⚠️ 商汤 | MLLM自动广告剪辑 | 技术储备 |
| P2 | [Skill-Virbo-Multilingual-Avatar-UGC](./Skill-Virbo-Multilingual-Avatar-UGC.md) | arXiv:2403.11700 | ⚠️ 商业产品 | 多语言批量UGC | 35-60万 |

## 三场景覆盖矩阵

| 场景 | 无人物 | 有虚拟人 | 品牌营销 | UGC口播 |
|------|--------|---------|---------|---------|
| 商品上架 | Phantom | AnchorCrafter | — | — |
| 品牌推广 | Aquarius | BrandFusion | Text-to-Edit | — |
| 电商UGC | — | Virbo | — | DAWN |

## 技术栈

| 模型/工具 | 定位 | License |
|-----------|------|---------|
| Wan2.2-A14B | 开源 SOTA T2V | Apache 2.0 |
| CogVideoX-5B | 最佳 DiT baseline | Apache 2.0 |
| Phantom-Wan-1.3B/14B | 主体一致性 I2V | Apache 2.0 |
| AnchorCrafter | 虚拟主播 HOI | 开源 |
| DAWN-pytorch | NAR Talking Head | 开源 |

## 统计数据

- 已萃取: 8
- 累计年化 ROI: 400-700 万元
