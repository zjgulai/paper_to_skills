# paper2skills Skill 关联图谱

## 技能总览

| 领域 | Skill | 功能 | 优先级 |
|------|-------|------|--------|
| 因果推断 | Uplift Modeling | 广告归因、促销效果评估 | ⭐⭐⭐⭐ |
| A/B实验 | Multi-Armed Bandit | 广告素材动态优化 | ⭐⭐⭐⭐⭐ |
| 时间序列 | Demand Forecasting | 销量预测、需求预测 | ⭐⭐⭐⭐⭐ |
| 供应链 | Inventory Optimization | 安全库存、备货策略 | ⭐⭐⭐⭐ |
| 推荐系统 | Matrix Factorization | 商品推荐、复购推荐 | ⭐⭐⭐⭐ |
| 增长模型 | Churn Prediction | 用户流失预警 | ⭐⭐⭐⭐ |

## 技能关联矩阵

```
                    Uplift   MAB    Forecast  Inventory  MatrixFac  Churn
                    Modeling                             Factorization
Uplift Modeling      -      ⭐      ⭐         ⭐           ⭐        ⭐
Multi-Armed Bandit   ⭐      -      ⭐                     ⭐
Demand Forecasting   ⭐      ⭐      -         ⭐⭐⭐        ⭐        ⭐
Inventory Opt        ⭐             ⭐⭐⭐       -           ⭐        ⭐
Matrix Factorization ⭐      ⭐      ⭐         ⭐           -        ⭐
Churn Prediction     ⭐      ⭐      ⭐         ⭐           ⭐        -
```

### 关联强度说明

- ⭐⭐⭐ 强关联：可直接组合，产生协同效应
- ⭐ 中等关联：可配合使用，需额外适配
- 空：无直接关联

## 关联逻辑说明

### 1. 预测类技能（Demand Forecasting）

- **依赖数据**: 历史销量、促销活动、季节性因素
- **输出**: 未来销量预测
- **下游技能**: Inventory Optimization（依赖预测结果）
- **横向关联**: Churn Prediction（预测维度可扩展）

### 2. 库存优化（Inventory Optimization）

- **依赖输入**: Demand Forecasting 的预测结果
- **核心决策**: 安全库存、补货量、补货时机
- **输出**: 库存策略
- **上游依赖**: Demand Forecasting（必须先有预测）

### 3. 因果推断（Uplift Modeling）

- **核心问题**: 区分"自然转化"和"营销增量"
- **应用场景**: 广告归因、促销效果评估、用户分层
- **下游技能**: Multi-Armed Bandit（MAB需要真实增量信号）

### 4. 探索与优化（Multi-Armed Bandit）

- **核心问题**: 探索-利用权衡
- **应用场景**: 广告素材测试、定价策略实验
- **上游依赖**: Uplift Modeling（需要真实效果评估）
- **横向关联**: Matrix Factorization（可扩展到推荐场景）

### 5. 推荐系统（Matrix Factorization）

- **核心问题**: 用户-商品匹配
- **应用场景**: 商品推荐、搜索排序、首页推荐
- **横向关联**: Churn Prediction（推荐可作为挽留手段）

### 6. 用户增长（Churn Prediction）

- **核心问题**: 识别流失风险用户
- **应用场景**: 流失预警、用户生命周期管理
- **下游技能**: Matrix Factorization（推荐挽留）

## 技能树结构

```
                         ┌─────────────────────────────────────┐
                         │         技能树 (Skill Tree)          │
                         └─────────────────────────────────────┘

    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
    │  基础技能    │      │  核心技能    │      │  组合技能    │
    └─────────────┘      └─────────────┘      └─────────────┘

入门 ──┬── 基础统计 (概率分布、假设检验)
      ├── SQL查询 (数据提取)
      └── Python数据分析 (pandas, numpy, visualization)

      │
      ▼
基础 ──┬── Demand Forecasting (需求预测基础)
      │      │
      │      └──> Inventory Optimization (依赖预测)
      │
      ├── Uplift Modeling (因果推断基础)
      │      │
      │      └──> Multi-Armed Bandit (依赖增量信号)
      │
      └── Churn Prediction (用户分析基础)
             │
             └──> Matrix Factorization (依赖用户理解)

      │
      ▼
进阶 ──┬── Inventory Optimization
      ├── Multi-Armed Bandit
      └── Matrix Factorization

      │
      ▼
组合 ──┬── 供应链优化 = Demand Forecasting + Inventory Optimization
      ├── 广告优化 = Multi-Armed Bandit + Uplift Modeling
      └── 用户运营 = Churn Prediction + Matrix Factorization
```

## 技能组合推荐

### 组合一：广告投放优化（高优先级）
- **Multi-Armed Bandit** + **Uplift Modeling**
- 效果：MAB 动态分配流量，Uplift 评估真实增量效果
- 场景：Facebook/Google 广告素材测试
- 依赖关系：MAB 依赖 Uplift 提供的真实增量信号

### 组合二：供应链优化（高优先级）
- **Demand Forecasting** + **Inventory Optimization**
- 效果：预测驱动库存决策，降低缺货和积压
- 场景：海外仓备货
- 依赖关系：Inventory Optimization 必须依赖 Demand Forecasting 输出

### 组合三：用户运营（中等优先级）
- **Churn Prediction** + **Matrix Factorization**
- 效果：识别流失风险 + 个性化推荐挽留
- 场景：沉默用户激活
- 依赖关系：Churn Prediction 输出高风险用户 → Matrix Factorization 做个性化推荐

### 组合四：全链路优化（终极目标）
- **Demand Forecasting** → **Inventory Optimization**
- **Churn Prediction** + **Matrix Factorization** → 个性化推荐
- **Uplift Modeling** → 营销活动效果评估
- **Multi-Armed Bandit** → 实时素材优化

## 应用优先级

### Phase 1: 立即可落地（1-2周）
1. **Multi-Armed Bandit** - 广告素材测试
2. **Demand Forecasting** - 销量预测
3. **Multi-Armed Bandit** - 落地页/文案测试

### Phase 2: 短期可落地（2-4周）
4. **Inventory Optimization** - 库存策略
5. **Uplift Modeling** - 归因分析
6. **Churn Prediction** - 流失预警

### Phase 3: 中期规划（1-2月）
7. **Matrix Factorization** - 推荐系统
8. **组合应用** - 端到端优化

## 技能依赖关系

```
入门技能：
├── 基础统计
├── SQL查询
└── Python数据分析

基础技能：
├── Demand Forecasting (需求预测基础)
├── Churn Prediction (用户分析基础)
└── Uplift Modeling (因果推断基础)

进阶技能：
├── Inventory Optimization (依赖 Demand Forecasting)
├── Multi-Armed Bandit (依赖 基础统计)
└── Matrix Factorization (依赖 基础机器学习)

组合技能：
├── 供应链优化 = Demand Forecasting + Inventory Optimization
├── 广告优化 = Multi-Armed Bandit + Uplift Modeling
└── 用户运营 = Churn Prediction + Matrix Factorization
```
