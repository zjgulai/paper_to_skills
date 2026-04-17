# 论文阅读笔记: Transfer Learning for Contextual Joint Assortment-Pricing under Cross-Market Heterogeneity

## 基本信息
- **arXiv ID**: 2603.18114
- **标题**: Transfer Learning for Contextual Joint Assortment-Pricing under Cross-Market Heterogeneity
- **作者**: Elynn Chen, Xi Chen, Yi Zhang
- **发表时间**: 2026-03-18
- **机构**: NYU Stern, Tsinghua SEM

## 核心贡献
提出 TJAP (Transfer Joint Assortment-Pricing) 框架，首次在跨市场异质性环境下实现上下文联合选品定价的迁移学习。通过结构化效用偏移模型，将共享偏好方向与异质性方向分离，实现"聚合降方差、去偏控偏差"的 bias-aware 迁移决策。

## 算法流程（3个核心组件）
1. **Aggregate-then-Debias 估计**: 先用所有源市场数据聚合估计共享偏好结构，再用目标市场数据通过 ℓ₁-正则化修正稀疏偏移
2. **Two-Radius 乐观决策**: 构建同时包含统计不确定半径（variance radius，随聚合数据收缩）和迁移偏差半径（transfer-bias radius，与源市场数量无关）的UCB-style决策规则
3. **Episodic Information-Geometry Control**: 以几何递增的episode冻结信息几何结构，仅在目标市场曲率不足时触发强制探索

## 数学直觉
- **Contextual MNL 效用模型**: v_it^(h) = ⟨x_it^(h), θ^(h)⟩ - ⟨x_it^(h), γ^(h)⟩ p_it^(h)
- **结构化偏好偏移**: 源市场与目标市场的参数差异集中在最多 s₀ 个坐标上（稀疏偏移）
- **后悔界**: Regret = Õ(d√(T/(1+H)) + s₀√T)
  - 第一项: 共享偏好方向的方差缩减，源市场数量 H 越多效果越好
  - 第二项: 异质性方向的不可约适配成本，仅依赖目标市场数据

## 实验结果
- **合成数据**: d∈{10,20,50}, N∈{30,100}, K=5, H∈{0,1,3,5}, T=2000
- **主要发现**:
  1. 增加源市场数量 H 显著降低累积后悔（与 1/√(1+H) 一致）
  2. TJAP 一致优于单市场基线 CAP
  3. 无去偏的 naive pooling (POOL(H)) 被 TJAP 均匀支配，且 s₀ 越大差距越明显
  4. 联合建模 assortment + pricing 显著优于仅定价方法 (M3P, ONS-MPP)

## 母婴出海适配点
- Momcozy 在美国（成熟市场，数据丰富）和德国（目标市场，数据稀疏）销售吸奶器、消毒器、暖奶器
- 通过 TJAP 将美国市场的历史购买数据迁移到德国市场，同时纠正德美两国用户在价格敏感度、功能偏好等维度上的稀疏差异
- 解决新品进入新市场时"数据少、决策难"的痛点，将单市场试错周期从数月缩短到数周
