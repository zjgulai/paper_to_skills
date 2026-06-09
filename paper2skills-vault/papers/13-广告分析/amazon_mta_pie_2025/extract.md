# 萃取记录: Amazon Ads Multi-Touch Attribution (PIE Framework)

## 论文信息

- **来源**: Amazon Ads / arXiv 2508.08209
- **标题**: Amazon Ads Multi-Touch Attribution & Predicted Incrementality by Experimentation (PIE)
- **发布时间**: 2025
- **领域**: 13-广告分析
- **研究方向**: 多触点归因 (MTA), 实验增量校准 (Experimental Calibration), 机器学习归因 (ML Attribution)

## 核心算法提炼

### 算法名称
Predicted Incrementality by Experimentation (PIE) / 实验增量预测归因框架

### 核心思想
广告归因存在一个无解的悖论（The Core Dilemma）：
- 纯跑 A/B 测试（RCTs）：得出的增量绝对真实无偏差，但它只能算总盘子，无法下钻到具体的点击（Touchpoint）级别，颗粒度太粗。
- 纯跑机器学习 MTA（如 Shapley, Attention模型）：能精确算到每一个点击的贡献，但由于“选择偏差”（平台总是把广告投给容易买的人），算出来的转化全是水军效应，被严重高估。

Amazon 在 2025 年彻底颠覆了这个悖论，提出了 **PIE (Predicted Incrementality by Experimentation) 融合框架**：
1. **建立 Ground Truth 锚点**：首先在底层运行大规模随机对照实验（RCT），获得绝对干净无偏的“大盘总因果增量”。
2. **训练因果校准模型（Causal Calibration Model）**：用深度学习模型去吃海量的历史特征（包括那些断点数据、不完整的触点），但它的目标不是去预测“用户买不买”，而是去预测**“这个用户在 RCT 实验中表现出的增量敏感度”**。
3. **分配归因份额（Attribution Shares）**：把机器学习算出的微观触点概率（高精度的有偏数据），强制用 RCT 的宏观增量（低精度的无偏数据）进行重新缩放（Rescaling / Calibration）。使得所有底层微观触点的总贡献，严丝合缝地等于大盘的真实因果增量。

### 为什么好用（优势）
1. **完美结合 ML 的颗粒度与 RCT 的可信度**：连最严苛的财务和审计都能挑不出毛病（有实验背书），同时又能满足一线优化师的需求（能看清每个广告组的贡献）。
2. **天然抗数据缺失**：即使遇到苹果 ATT 导致的数据断点，由于顶层有 RCT 实验的“总盘子”镇压，ML 模型在分配触点权重时不会发生灾难性的偏离。

## 业务适配设计：大型站内外全渠道预算沙盘核准

### 场景: 品牌独立站 Google Search 与 Facebook 展现的真实触点分配
- **痛点**：老板只相信 A/B 实验的增量，但一线广告投手只能看平台的 Last-click 数据来调整单条计划出价。两边永远对不上账。
- **方案落地**：
  - 定期（每季度）跑一次 Geo-based 的 A/B 停投实验，获得“本季度在加州投入 10 万美金 Facebook 带来了真实 20 万 GMV 增量”的绝对锚点（RCT 阶段）。
  - 用日常的残缺多触点数据（包含点击和曝光）训练一个注意力神经网络。算出每条具体的 FB 视频和 Google 关键词的“疑似权重”。
  - 执行 PIE 校准：将所有微观广告组的疑似权重强行约束到那“20 万真实增量”的盘子里。
  - 最终，投手拿到的系统报表，每一条 Campaign 甚至每一个点击的归因价值，都是完全去偏的“真实增量收益”。
- **预期价值**：弥合高管层（看因果增量）与执行层（看触点归因）的数据鸿沟，打造亚马逊级别的下一代业财一体化营销衡量系统。