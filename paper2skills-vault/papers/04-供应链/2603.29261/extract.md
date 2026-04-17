# 萃取记录: Monodense DLM - 单品价格弹性估计

## 论文信息

- **arXiv ID**: 2603.29261
- **标题**: Monodense Deep Neural Model for Determining Item Price Elasticity
- **作者**: Lakshya Garg, Sai Yaswanth, Deep Narayan Mishra, Karthik Kumaran, Anupriya Sharma, Mayank Uniyal (Walmart Inc.)
- **萃取日期**: 2026-04-16
- **领域**: 04-供应链

## 核心算法提炼

### 算法名称
Monodense Deep Learning Model (Monodense-DLM) for Price Elasticity

### 核心思想
传统价格弹性估计依赖计量经济学方法（log-log OLS、AIDS 等），需要强函数形式假设且难以处理高维商品特征、季节性、竞品价格等复杂交互。Walmart 团队提出一种**无需对照实验（treatment-control free）**的深度学习方法，核心创新是 **Monodense 层**——通过在神经网络中强制价格与需求的单调递减关系（价格↓→需求↑），确保模型输出的弹性永远为负，符合经济学一致性。

### 数学直觉

1. **价格弹性定义**
   $$E_d = \frac{\% \text{ change in quantity demanded}}{\% \text{ change in price}}$$
   经济学直觉：需求对价格敏感，弹性通常为负。

2. **Monodense 层约束**
   对于输入特征 $x_i$ 设置单调性指示向量 $t_i$：
   - $t_i = 1$：单调递增，权重约束 $w_i \geq 0$
   - $t_i = -1$：单调递减，权重约束 $w_i \leq 0$（价格特征）
   - $t_i = 0$：无约束

   实现方式：在每次前向传播时对权重做投影
   $$w_i^{\text{eff}} = \begin{cases} \max(0, w_i) & t_i = 1 \\ \min(0, w_i) & t_i = -1 \\ w_i & t_i = 0 \end{cases}$$

3. **激活函数变体**
   基于凸激活函数 $\rho$（如 ReLU、ELU、SELU），构造三种神经元子集：
   - **原始凸激活**: $\rho(x)$
   - **凹上界激活**: $\tilde{\rho}(x) = -\rho(-x)$
   - **有界激活**: 
     $$\bar{\rho}(x) = \begin{cases} \rho(x+1) - \rho(1) & x < 0 \\ \rho(x-1) + \rho(1) & \text{otherwise} \end{cases}$$

4. **弹性评估公式**
   模型输出未来月份的需求量 $\hat{y}$。对给定价格变化 $\Delta p$，弹性为：
   $$\mathcal{E}_{\Delta p} = \frac{\hat{y}(p + \Delta p) - \hat{y}(p)}{\hat{y}(p)} \times \frac{p}{\Delta p}$$

5. **数据构造（Lead-Lag Cross Join）**
   - 将历史交易数据按月份聚合
   - 对每个商品做 lead month 和 lag month 的交叉组合（1-12个月差距）
   - 构造价格变化百分比 $S_t$、lag 月份的需求量 $D_t$ 等监督信号
   - 加入库存、商品属性、季节事件、竞品价格、替代品等上下文特征

### 关键假设
- 历史交易数据中存在自然的价格波动（无需人为实验）
- 价格与需求存在单调递减关系（经数据过滤后成立）
- 商品数量级可达百万级，模型需要可扩展
- 弹性估计可用于促销规划、动态定价、收入管理

## 业务适配设计

### 场景1: Momcozy 跨境电商单品动态定价
Momcozy 在 Amazon/Wayfair 等平台销售吸奶器、温奶器等母婴用品，SKU 数量庞大且面临激烈价格竞争。使用 Monodense-DLM：
1. 收集各平台历史销售数据（月销量、价格、促销标记、库存状态）
2. 加入竞品价格、季节性（Prime Day、黑五）、产品属性（容量、材质、功能）
3. 训练 Monodense-DLM 预测不同价格下的需求量
4. 计算每个 SKU 的价格弹性，识别高弹性商品（价格敏感）vs 低弹性商品（价格不敏感）
5. 对高弹性 SKU 采用竞争性低价策略，对低弹性 SKU 采用溢价策略，提升整体利润率

### 场景2: 季度促销弹性预判
大促前需要评估降价对销量的拉动效果。传统方法只能依赖历史同类型促销的回溯，Monodense-DLM 可以：
- 输入拟定的促销价格
- 直接输出预测销量和对应弹性
- 快速筛选出"降价能带来显著销量提升"的潜力 SKU，避免"降价不见量"的促销陷阱

## 代码实现

- **路径**: `paper2skills-code/supply_chain/monodense_elasticity/model.py`
- **设计**: 
  - 使用 PyTorch 实现 Monodense 层（带权重符号约束）
  - 实现三种激活变体（凸、凹、有界）
  - 构建简化的 DLM 网络预测需求量
  - 提供弹性计算接口和价格敏感性分析
  - 附带 Momcozy 母婴电商场景的合成数据演示
- **验证状态**: 待验证

## 性能基准（论文结果）

在 Walmart 的十亿级交易数据上，Monodense-DLM 相比 LGBM 和 Double Machine Learning：

| 模型 | WMAPE | MAE |
|------|-------|-----|
| Monodense-DLM | **30.9%** | **0.36** |
| LGBM | 35.9% | 0.42 |
| Double Machine Learning | 36.1% | 0.43 |

