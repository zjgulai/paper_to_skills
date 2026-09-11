# Multi-Armed Bandit 实施指南

## 场景
Facebook/Google 广告素材动态测试优化

## 技术架构

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  广告平台 API   │────▶│  MAB 决策引擎    │────▶│  流量分配服务   │
│ (Facebook/Google)   │  (Thompson Sampling) │  │  (实时返回素材)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
         ▲                        │
         │                        ▼
         │               ┌──────────────────┐
         │               │   数据仓库       │
         │               │ (用户行为数据)   │
         │               └──────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐     ┌──────────────────┐
│   效果监控      │◀────│   模型更新      │
│  (实时报表)    │     │   (每日重训)    │
└─────────────────┘     └──────────────────┘
```

## 快速开始

### 1. 数据接入

```python
from paper2skills_code.ab_testing.multi_armed_bandit import ThompsonSampling

# 初始化
bandit = ThompsonSampling(
    n_arms=4,
    arm_names=['creative_A', 'creative_B', 'creative_C', 'creative_D']
)

# 模拟数据流
for impression in ad_impressions:
    # 选择素材
    selected_creative = bandit.select_arm()

    # 展示给用户
    user_sees(selected_creative)

    # 记录转化
    if user_converts():
        bandit.update(selected_creative, reward=1)
    else:
        bandit.update(selected_creative, reward=0)
```

### 2. 生产部署

```python
# 部署配置
config = {
    'algorithm': 'thompson_sampling',
    'exploration_rate': 0.1,  # 10% 流量探索
    'min_samples': 100,  # 每个素材最少展示100次
    'auto_kill_threshold': 0.3,  # 相对CTR低于30%自动停投
}
```

## 参数调优

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| exploration_rate | 探索比例 | 10-20% |
| min_samples | 最小样本量 | 100-500 |
| auto_kill_threshold | 自动淘汰阈值 | 0.3-0.5 |
| update_frequency | 更新频率 | 实时 |

## 监控指标

- 各素材曝光量、点击量、转化量
- 流量分配比例
- 总转化率
- 预期 regret

## 注意事项

1. **数据质量**：确保转化数据准确回传
2. **延迟处理**：广告平台有延迟，需处理滞后
3. **冷启动**：新素材需要足够曝光才能决策
4. **伦理**：避免过度探索，保护用户体验
