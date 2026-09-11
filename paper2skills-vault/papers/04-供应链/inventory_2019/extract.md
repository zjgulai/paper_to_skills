# 论文信息

## Paper 2: Multi-Echelon Inventory Optimization

### 论文信息
- **标题**: Multi-Echelon Inventory Optimization for Supply Chain Management
- **领域**: 供应链 / 库存优化

### 核心算法
1. **Base Stock Policy**: 基础库存策略
2. **Newsvendor Model**: 报童模型
3. **(s, S) Policy**: (s, S) 库存策略

### 关键公式
- 最佳订购量: Q* = F⁻¹( (c_u) / (c_u + c_o) )
- 安全库存: SS = z × σ_L
- 期望缺货概率: P(stockout) = 1 - F(SS/σ)

---

# 论文摘要

We present a comprehensive framework for multi-echelon inventory optimization in retail supply chains. The key insight is to optimize inventory levels across the entire supply chain simultaneously, rather than at each node independently. We introduce several methods including base stock optimization, (s, S) policy approximation, and reinforcement learning-based dynamic ordering. Our methods account for lead times, demand uncertainty, and service level constraints. The framework is particularly effective for cross-border e-commerce where lead times are longer and demand variability is higher.
