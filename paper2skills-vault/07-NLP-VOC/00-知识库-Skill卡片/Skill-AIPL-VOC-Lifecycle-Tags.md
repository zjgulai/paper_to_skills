# Skill Card: AIPL 生命周期 × VOC 动态标签体系
# AIPL-VOC Lifecycle Dynamic Tagging

**原创整合**: 基于 AIPL 消费者行为模型 + VOC 文本分析技术栈
**理论基础**: 阿里 AIPL 模型 (Awareness-Interest-Purchase-Loyalty) + 方面情感分析
**适用领域**: 用户生命周期管理、精细化运营、动态分群、个性化触达

---

## ① 算法原理

### 核心思想
传统用户标签是静态的（如"90后宝妈"），无法反映购买决策过程中的动态变化。AIPL × VOC 体系将用户 VOC 文本（搜索、咨询、评论、分享）映射到生命周期阶段，实现**阶段识别 → 迁移检测 → 标签演化**的闭环。

### AIPL 四阶段定义

| 阶段 | 英文 | 业务含义 | VOC 信号特征 |
|------|------|----------|-------------|
| A | Awareness (认知) | 首次接触品牌/产品 | 搜索关键词、浏览行为、初步咨询 |
| I | Interest (兴趣) | 加购对比、深入了解 | 加购物车、查看评价、价格咨询 |
| P | Purchase (购买) | 完成首购 | 下单评价、开箱反馈、售后咨询 |
| L | Loyalty (忠诚) | 复购推荐 | 回购评论、社交分享、会员活跃 |

### 技术架构

**三层模型**：

```
VOC 信号层 (文本输入)
    ├── 搜索关键词
    ├── 浏览/点击行为描述
    ├── 客服咨询记录
    ├── 产品评论文本
    ├── 社交分享内容
    └── 主动反馈/投诉
           │
           ▼
    阶段分类层 (AIPLClassifier)
    ├── 关键词匹配打分
    ├── 信号类型加权
    ├── 时间衰减权重
    └── 阶段概率归一化
           │
           ▼
    迁移检测层 (StageTransitionDetector)
    ├── 迁移触发词识别
    ├── 信号密度分析
    ├── 情感极性判断
    └── 置信度计算
           │
           ▼
    动态标签层 (DynamicTagSystem)
    ├── 生命周期阶段标签
    ├── 运营策略标签
    ├── 触达渠道标签
    ├── 迁移预警标签
    └── 方面关注标签
```

### 阶段分类算法

对用户的 VOC 信号序列 $S = \{s_1, s_2, ..., s_n\}$，计算各阶段得分：

$$score_{stage} = \sum_{s \in S} \sum_{k \in K_{stage}} \mathbb{1}[k \in s.text] \times (1 + |s.sentiment|) \times w_{time}(s)$$

其中 $K_{stage}$ 是阶段关键词词典，$w_{time}$ 是时间衰减权重（近期信号权重更高）。

最终阶段：

$$stage_{current} = \arg\max_{stage} \frac{score_{stage}}{\sum_j score_j}$$

### 迁移检测算法

检测用户是否处于阶段迁移临界点：

1. **触发词匹配**：检测最近信号中是否包含迁移触发词（如"加购→下单→付款"）
2. **信号密度**：短时间内多条相关信号 = 高迁移意愿
3. **情感加速**：正向情感加速迁移，负向情感阻碍迁移

迁移置信度：

$$confidence = \min\left(\frac{\sum_{s \in S_{recent}} \sum_{t \in T} \mathbb{1}[t \in s.text] \times (1 + \max(0, s.sentiment))}{2}, 1.0\right)$$

---

## ② 母婴出海应用案例

### 场景1：新品冷启动期用户培育

**业务问题**
母婴品牌新进入东南亚市场，预算有限，需要精准识别哪些"认知期"用户值得投入培育成本，哪些已进入"兴趣期"可以促单。

**数据输入**
- 搜索日志关键词
- 社媒互动文本
- 客服咨询记录
- 站内浏览行为描述

**分析流程**
1. 对所有潜在用户运行 AIPL 阶段分类
2. 识别出 1200 名"认知期"用户和 800 名"兴趣期"用户
3. 对"兴趣期"用户检测迁移信号：发现 300 名用户有强烈购买意愿（置信度 > 0.5）
4. 输出差异化策略：
   - 认知期（1200人）：内容种草 + KOL测评视频推送
   - 兴趣期-高意愿（300人）：限时优惠券 + 免邮活动
   - 兴趣期-一般（500人）：评价引导 + 对比内容

**预期效果**
- 营销预算精准度提升 40%
- 兴趣期到购买期转化率从 8% 提升至 15%

### 场景2：大促前用户状态盘点

**业务问题**
黑五大促前一周，需要知道当前用户池中各 AIPL 阶段分布，以便制定分阶段触达策略。

**分析流程**
1. 全量用户 VOC 信号刷新
2. 输出分群报告：
   - A 认知：45% → 策略：品牌广告覆盖
   - I 兴趣：30% → 策略：预售锁客 + 定金膨胀
   - P 购买：20% → 策略：复购提醒 + 凑单推荐
   - L 忠诚：5% → 策略：老客专享 + 裂变激励
3. 检测迁移就绪用户（即将进入下一阶段），提前干预

---

## ③ 代码模板

核心模块：`paper2skills-code/nlp_voc/aipl_voc_lifecycle/model.py`

```python
from aipl_voc_lifecycle import (
    DynamicTagSystem,
    AIPLStage,
    VOCSignal,
    VOCSignalType,
    run_aipl_voc_analysis,
)

# 方式1: 快速运行（使用合成数据）
result = run_aipl_voc_analysis(n_users=1000)
print(result["segment_report"])

# 方式2: 使用真实 VOC 信号
signals = [
    VOCSignal(
        user_id="u001",
        text="婴儿纸尿裤推荐哪个牌子",
        signal_type=VOCSignalType.SEARCH,
        timestamp=datetime.now(),
        sentiment=0.0,
    ),
    VOCSignal(
        user_id="u001",
        text="加购物车等活动降价",
        signal_type=VOCSignalType.BROWSE,
        timestamp=datetime.now(),
        sentiment=0.2,
    ),
]

system = DynamicTagSystem()
system.ingest(signals)
profile = system.get_profile("u001")
print(f"阶段: {profile.current_stage.value}")
print(f"标签: {profile.tags}")

# 分群报告
report = system.get_segment_report()
```

**扩展方向**
- 接入真实 ABSA 模型替换规则版情感分析（见 Skill-ABSA-BERT-MoE）
- 接入 NPS 驱动因素分析识别忠诚用户痛点（见 Skill-NPS-Driver-Analysis）
- 结合 OfflineRL 优化触达时机（见 Skill-OfflineRL-触达时机优化）

---

## ④ 技能关联

### 前置技能
| 技能 | 关系 | 说明 |
|------|------|------|
| ABSA-BERT-MoE | 依赖 | 提供方面级情感提取 |
| NPS Driver Analysis | 依赖 | 忠诚期用户满意度归因 |
| CSK-Customer-Sentiment-Clustering | 组合 | 先聚类再分阶段 |

### 扩展技能
| 技能 | 关系 | 说明 |
|------|------|------|
| PERSONABOT-RAG用户画像 | 上游 | VOC 信号来源之一 |
| OfflineRL-触达时机优化 | 下游 | 阶段标签驱动触达时机 |
| TSCAN-上下文感知挽回 | 下游 | 购买期→忠诚期迁移失败时挽回 |
| DQN-Purchase-Prediction | 互补 | 行为数据角度预测购买意图 |
| User-Lifecycle-STAN | 互补 | 连续时间角度建模生命周期 |

---

## ⑤ 业务价值评估

| 维度 | 评分 | 说明 |
|------|------|------|
| ROI潜力 | ★★★★★ | 精准分群可使营销转化率提升 30-50% |
| 实施难度 | ★★☆☆☆ | 规则版仅需关键词词典；生产环境需 VOC 数据管道 |
| 数据需求 | ★★★☆☆ | 需要搜索/浏览/咨询/评论等多源 VOC 数据 |
| 可解释性 | ★★★★★ | 关键词匹配逻辑透明，业务可直接理解 |
| 时效性 | ★★★★★ | 实时/准实时更新，支撑运营即时决策 |

**综合评分: 9/10**

**核心优势**：打通"用户说了什么"到"用户在哪个阶段"到"该用什么策略触达"的完整链路。

---

## 参考资源

- 代码目录: `paper2skills-code/nlp_voc/aipl_voc_lifecycle/`
- 理论基础: 阿里 AIPL 消费者运营模型
- 关联论文: Boughamni et al. (2025) "From Reviews to Actionable Insights" — 方面提取方法论
