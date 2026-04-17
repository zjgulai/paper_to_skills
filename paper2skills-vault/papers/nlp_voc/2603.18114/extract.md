# 萃取记录: TJAP - 跨市场品类组合定价

## 论文信息
- **arXiv ID**: 2603.18114
- **标题**: Transfer Learning for Contextual Joint Assortment-Pricing under Cross-Market Heterogeneity
- **作者**: Elynn Chen, Xi Chen, Yi Zhang
- **发表时间**: 2026-03-18
- **领域**: 07-NLP-VOC (品类管理与分析延伸)

## 核心算法提炼

### 算法名称
TJAP (Transfer Joint Assortment-Pricing)

### 核心思想
将多市场历史销售数据安全迁移到新市场的联合选品与定价决策中。通过"聚合降方差、去偏控偏差"的 bias-aware 策略，在利用源市场丰富数据的同时，隔离跨市场偏好差异带来的系统性偏差。

### 数学直觉
1. **Contextual MNL 效用模型**: $v_{it}^{(h)} = \langle x_{it}^{(h)}, \theta^{(h)} \rangle - \langle x_{it}^{(h)}, \gamma^{(h)} \rangle p_{it}^{(h)}$
2. **结构化偏好偏移**: 源市场与目标市场的参数差异集中在最多 $s_0$ 个稀疏坐标上
3. **Aggregate-then-Debias**: 先池化源市场数据估计共享结构，再用目标市场数据通过 L1 正则化修正稀疏偏移
4. **Two-Radius UCB**: 同时考虑统计不确定半径和迁移偏差半径的乐观决策
5. **后悔界**: $\tilde{O}(d\sqrt{T/(1+H)} + s_0\sqrt{T})$

### 关键假设
- 跨市场差异具有稀疏结构（$s_0 \ll d$）
- 价格敏感度为正，保证最优价格有界
- 市场间上下文特征分布可比（或通过重要性加权修正）

## 业务适配设计

### 场景1：Momcozy 德国站新品上市选品定价
将美国站的丰富历史数据迁移到德国站，同时修正德美两国用户在静音性、价格敏感度等维度上的差异，将新市场试错周期从数月缩短到数周。

### 场景2：Momcozy 多平台差异化运营（Amazon US vs Temu US）
基于 Amazon 数据快速优化 Temu 的产品组合和定价策略，识别平台客群差异，实现智能化的差异化运营。

## 代码实现
- **路径**: `paper2skills-code/nlp_voc/tjap_cross_market_assortment_pricing/model.py`
- **设计**: 用线性概率模型近似 MNL MLE，用坐标下降软阈值实现 L1 去偏，用贪心启发式+网格搜索求解带容量约束的 assortment-pricing 问题
- **验证状态**: ✅ 通过
