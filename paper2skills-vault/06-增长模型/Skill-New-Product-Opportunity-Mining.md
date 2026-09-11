# Skill Card: New Product Opportunity Mining (新品机会挖掘模型)

---

## ① 算法原理

### 核心思想

新品机会挖掘模型解决的核心问题是：**在新商品上市前预测其成功概率，从而优化选品决策和资源配置**。与传统的事后分析不同，该模型通过多维度评估框架，在投入大量资源前识别高潜力新品。

该框架源自 [SSFF (Startup Success Forecasting Framework)](https://arxiv.org/abs/2405.19456) 研究，将创业成功预测方法论迁移到电商新品场景。

### 数学直觉

**LLM-Enhanced Random Forest (LLM-RF)**:

$$
P(success|X) = \frac{1}{M} \sum_{m=1}^{M} I(h_m(X) = 1)
$$

其中 $h_m$ 是第 $m$ 棵决策树，$X = \{x_1, x_2, ..., x_{14}\}$ 是14个维度的特征向量。

**关键创新 - 类别特征处理**:
- 传统Random Forest难以处理高基数类别特征
- LLM将类别特征转换为语义嵌入：$e_i = \text{LLM}_{\text{embed}}(c_i)$
- 通过GPT-4o进行智能分类，将非结构化文本转化为结构化维度

**Founder-Idea Fit Score (FIF) 迁移到 Product-Market Fit**:

$$
PMF(P, M) = (6-L_P) \times O_M - L_P \times (1-O_M)
$$

- $L_P$: 产品成熟度等级 (1-5)
- $O_M$: 市场机会得分 (0-1)
- 归一化到 $[-1, 1]$，1为最佳契合

**余弦相似度计算产品-市场匹配**:

$$
\text{similarity}(P, M) = \frac{\vec{e}_P \cdot \vec{e}_M}{||\vec{e}_P|| \times ||\vec{e}_M||}
$$

其中 $\vec{e}_P$ 和 $\vec{e}_M$ 分别是产品和市场的文本嵌入向量。

### 18维度评估框架

| 维度类别 | 具体维度 | 评估内容 |
|---------|---------|---------|
| **市场维度** | 市场规模 | TAM/SAM/SOM评估 |
| | 市场增长率 | 年复合增长率(CAGR) |
| | 行业趋势 | 上升/平稳/下降 |
| | 监管环境 | 合规难度、认证要求 |
| **产品维度** | 产品创新度 | 技术/功能创新程度 |
| | 产品-市场契合 | 解决痛点精准度 |
| | 可扩展性 | 供应链复制能力 |
| | 用户参与度 | 预期复购率 |
| **执行维度** | 开发速度 | 上市时间预估 |
| | 团队专业度 | 母婴品类经验 |
| | 资金充足度 | 推广预算规模 |
| | 供应链稳定性 | 供应商可靠性 |
| **时机维度** | 入场时机 | 早/正当时/晚 |
| | 竞争强度 | 竞品数量与实力 |
| | 季节因素 | 母婴产品季节性 |
| | 平台政策 | 亚马逊/独立站政策 |
| | IP专利 | 知识产权保护 |
| | 投资背书 | 是否有头部供应商支持 |

### 关键假设

- **历史可预测未来**：过去成功的新品模式可以预测未来
- **多维度互补**：单一维度不足以预测，需综合评估
- **LLM理解语义**：GPT-4o能准确理解产品和市场描述
- **数据质量**：输入的18维度信息准确完整

---

## ② 吸奶器出海应用案例

### 场景一：智能穿戴式吸奶器新品评估

**业务问题**：
某供应商计划推出一款智能穿戴式吸奶器（可连接APP追踪奶量），需要评估是否值得投入大量营销资源，以及优先投放到哪个市场（美国/欧洲/东南亚）。

**数据要求**：
- 市场数据：各区域穿戴式吸奶器搜索量、竞品数量、价格带分布
- 产品数据：功能清单、专利情况、预计成本价、认证状态（FDA/CE）
- 团队数据：母婴品类运营经验、供应链稳定性评分
- 时机数据：竞品上市时间、平台政策变化（如亚马逊对新品的扶持期）

**预期产出**：
- 成功概率评分 (0-100%)
- 三个维度的细分得分：市场可行性/产品可行性/执行能力
- 区域优先级排序（美国 vs 欧洲 vs 东南亚）
- 投资建议：全力投入/试点测试/暂缓观望
- 关键风险提示

**业务价值**：
- 避免盲目投入：预计节省无效营销费用 30-50%
- 精准资源配置：将预算集中于高潜力新品
- 快速止损：识别低潜力产品，提前调整策略
- ROI估算：假设年新品投入500万，准确率77%可优化决策质量，预计提升ROI 15-25%

---

### 场景二：辅食机配件新品线扩展决策

**业务问题**：
现有辅食机热销，考虑扩展配件产品线（专用蒸笼、便携盒、清洁刷套装）。需要评估哪个配件最值得优先开发，以及预计市场接受度。

**数据要求**：
- 用户需求数据：评论分析中提到的配件需求、客服咨询记录
- 竞品数据：竞品配件销量、评价、价格
- 产品数据：各配件的开发成本、生产周期、预计定价
- 协同数据：与主产品的搭配购买率预期

**预期产出**：
- 各配件的成功概率排名
- 产品-市场契合度评分
- 最优上市时间窗口
- 预期首月销量区间

**业务价值**：
- 产品线优化：聚焦高潜力配件，避免sku膨胀
- 库存风险控制：按预测销量控制首批库存
- 交叉销售提升：预计配件可提升主产品转化率 10-15%

---

## ③ 代码模板

```python
"""
New Product Opportunity Mining Model
新品机会挖掘模型 - 基于SSFF框架
用于母婴出海电商新品成功预测
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import openai
from typing import Dict, List, Tuple, Optional
import json
import warnings
warnings.filterwarnings('ignore')


class ProductOpportunityMiner:
    """新品机会挖掘模型"""

    # 18维度评估框架定义
    DIMENSIONS = {
        'market': ['market_size', 'growth_rate', 'industry_trend', 'regulatory_env'],
        'product': ['innovation_level', 'product_market_fit', 'scalability', 'user_engagement'],
        'execution': ['development_speed', 'team_expertise', 'funding_status', 'supply_chain'],
        'timing': ['market_timing', 'competition_intensity', 'seasonality',
                   'platform_policy', 'ip_protection', 'investor_backing']
    }

    def __init__(self, use_llm: bool = True, openai_api_key: Optional[str] = None):
        """
        初始化新品机会挖掘模型

        Args:
            use_llm: 是否使用LLM进行特征提取
            openai_api_key: OpenAI API密钥
        """
        self.use_llm = use_llm
        self.openai_api_key = openai_api_key
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.is_fitted = False
        self.dimension_weights = None

    def extract_dimensions_with_llm(self, product_description: str,
                                   market_context: str) -> Dict[str, str]:
        """
        使用LLM从文本中提取18维度评估

        Args:
            product_description: 产品描述
            market_context: 市场环境描述

        Returns:
            18维度的评估结果字典
        """
        if not self.use_llm:
            return self._manual_dimension_input()

        prompt = f"""
        请作为电商选品专家，基于以下产品信息，评估该新品在母婴出海电商领域的成功潜力。

        产品描述：{product_description}
        市场环境：{market_context}

        请对以下18个维度进行评估（输出JSON格式）：

        市场维度：
        1. market_size: 市场规模 [Large/Medium/Small]
        2. growth_rate: 市场增长率 [High/Medium/Low]
        3. industry_trend: 行业趋势 [Rising/Stable/Declining]
        4. regulatory_env: 监管环境 [Easy/Moderate/Difficult]

        产品维度：
        5. innovation_level: 产品创新度 [High/Medium/Low]
        6. product_market_fit: 产品市场契合 [Strong/Moderate/Weak]
        7. scalability: 可扩展性 [High/Medium/Low]
        8. user_engagement: 用户参与度预期 [High/Medium/Low]

        执行维度：
        9. development_speed: 开发速度 [Fast/Medium/Slow]
        10. team_expertise: 团队专业度 [High/Medium/Low]
        11. funding_status: 资金充足度 [Sufficient/Adequate/Limited]
        12. supply_chain: 供应链稳定性 [Stable/Moderate/Unstable]

        时机维度：
        13. market_timing: 入场时机 [Early/Just_Right/Late]
        14. competition_intensity: 竞争强度 [High/Medium/Low]
        15. seasonality: 季节因素 [Favorable/Neutral/Unfavorable]
        16. platform_policy: 平台政策 [Supportive/Neutral/Restrictive]
        17. ip_protection: IP保护 [Strong/Moderate/Weak]
        18. investor_backing: 供应商/投资方支持 [Strong/Moderate/Weak]

        只输出JSON格式结果，不要其他解释。
        """

        try:
            # 模拟LLM响应（实际使用时调用OpenAI API）
            # response = openai.ChatCompletion.create(...)
            # 这里返回模拟结果用于演示
            return self._simulate_llm_response()
        except Exception as e:
            print(f"LLM提取失败: {e}，使用默认评估")
            return self._simulate_llm_response()

    def _simulate_llm_response(self) -> Dict[str, str]:
        """模拟LLM响应（实际使用时替换为真实API调用）"""
        return {
            'market_size': 'Large',
            'growth_rate': 'High',
            'industry_trend': 'Rising',
            'regulatory_env': 'Moderate',
            'innovation_level': 'High',
            'product_market_fit': 'Strong',
            'scalability': 'High',
            'user_engagement': 'High',
            'development_speed': 'Medium',
            'team_expertise': 'High',
            'funding_status': 'Sufficient',
            'supply_chain': 'Stable',
            'market_timing': 'Just_Right',
            'competition_intensity': 'Medium',
            'seasonality': 'Favorable',
            'platform_policy': 'Supportive',
            'ip_protection': 'Moderate',
            'investor_backing': 'Strong'
        }

    def _manual_dimension_input(self) -> Dict[str, str]:
        """手动输入维度评估"""
        return {
            'market_size': 'Medium',
            'growth_rate': 'Medium',
            'industry_trend': 'Stable',
            'regulatory_env': 'Easy',
            'innovation_level': 'Medium',
            'product_market_fit': 'Moderate',
            'scalability': 'Medium',
            'user_engagement': 'Medium',
            'development_speed': 'Medium',
            'team_expertise': 'Medium',
            'funding_status': 'Adequate',
            'supply_chain': 'Stable',
            'market_timing': 'Just_Right',
            'competition_intensity': 'High',
            'seasonality': 'Neutral',
            'platform_policy': 'Neutral',
            'ip_protection': 'Weak',
            'investor_backing': 'Moderate'
        }

    def encode_dimensions(self, dimensions: Dict[str, str]) -> np.ndarray:
        """
        将维度评估编码为数值特征

        编码规则：
        - High/Large/Strong/Rising/Favorable/Supportive/Early/Fast/Sufficient/Stable/Just_Right = 1.0
        - Medium/Moderate/Neutral = 0.5
        - Low/Small/Weak/Declining/Unfavorable/Restrictive/Late/Slow/Limited/Unstable = 0.0
        """
        high_values = {'High', 'Large', 'Strong', 'Rising', 'Favorable',
                      'Supportive', 'Early', 'Fast', 'Sufficient', 'Stable', 'Just_Right'}
        low_values = {'Low', 'Small', 'Weak', 'Declining', 'Unfavorable',
                     'Restrictive', 'Late', 'Slow', 'Limited', 'Unstable'}

        encoded = []
        for dim_value in dimensions.values():
            if dim_value in high_values:
                encoded.append(1.0)
            elif dim_value in low_values:
                encoded.append(0.0)
            else:  # Medium/Moderate/Neutral
                encoded.append(0.5)

        return np.array(encoded)

    def calculate_product_market_fit(self, product_desc: str,
                                     market_desc: str) -> float:
        """
        计算产品-市场契合度分数 (PMF Score)

        使用文本相似度作为代理指标
        实际应用中可以使用预训练的语言模型嵌入
        """
        # 简化的PMF计算（实际使用text-embedding-3-large）
        product_keywords = set(product_desc.lower().split())
        market_keywords = set(market_desc.lower().split())

        if not product_keywords or not market_keywords:
            return 0.5

        intersection = product_keywords & market_keywords
        union = product_keywords | market_keywords

        jaccard_similarity = len(intersection) / len(union)

        # 映射到[-1, 1]范围
        pmf_score = 2 * jaccard_similarity - 1
        return pmf_score

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ProductOpportunityMiner':
        """
        训练模型

        Args:
            X: 特征矩阵 (n_samples, 18)
            y: 标签 (1=成功, 0=失败)
        """
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 训练Random Forest
        self.rf_model.fit(X_train, y_train)
        self.is_fitted = True

        # 评估模型
        y_pred = self.rf_model.predict(X_test)
        y_prob = self.rf_model.predict_proba(X_test)[:, 1]

        print("=" * 60)
        print("模型训练完成")
        print("=" * 60)
        print(f"\nAUC Score: {roc_auc_score(y_test, y_prob):.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, target_names=['失败', '成功']))

        # 计算维度重要性
        self._calculate_dimension_importance()

        return self

    def _calculate_dimension_importance(self):
        """计算各维度的重要性"""
        importance = self.rf_model.feature_importances_

        all_dims = []
        for dims in self.DIMENSIONS.values():
            all_dims.extend(dims)

        self.dimension_weights = dict(zip(all_dims, importance))

        print("\n维度重要性 (Top 10):")
        sorted_importance = sorted(
            self.dimension_weights.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        for dim, weight in sorted_importance:
            print(f"  {dim}: {weight:.4f}")

    def predict(self, dimensions: Dict[str, str]) -> Dict[str, any]:
        """
        预测新品成功概率

        Args:
            dimensions: 18维度评估字典

        Returns:
            包含预测结果的字典
        """
        if not self.is_fitted:
            raise ValueError("模型必须先训练！")

        # 编码维度
        X = self.encode_dimensions(dimensions).reshape(1, -1)

        # 预测
        success_prob = self.rf_model.predict_proba(X)[0, 1]
        prediction = self.rf_model.predict(X)[0]

        # 计算分类维度得分
        category_scores = self._calculate_category_scores(dimensions)

        # 生成建议
        recommendation = self._generate_recommendation(success_prob, category_scores)

        return {
            'success_probability': success_prob,
            'prediction': '成功' if prediction == 1 else '失败',
            'category_scores': category_scores,
            'recommendation': recommendation,
            'risk_level': self._assess_risk_level(success_prob),
            'dimension_assessment': dimensions
        }

    def _calculate_category_scores(self, dimensions: Dict[str, str]) -> Dict[str, float]:
        """计算各维度类别得分"""
        scores = {}

        for category, dims in self.DIMENSIONS.items():
            category_values = []
            for dim in dims:
                value = dimensions.get(dim, 'Medium')
                if value in {'High', 'Large', 'Strong', 'Rising', 'Favorable',
                           'Supportive', 'Early', 'Fast', 'Sufficient', 'Stable', 'Just_Right'}:
                    category_values.append(1.0)
                elif value in {'Low', 'Small', 'Weak', 'Declining', 'Unfavorable',
                              'Restrictive', 'Late', 'Slow', 'Limited', 'Unstable'}:
                    category_values.append(0.0)
                else:
                    category_values.append(0.5)

            scores[category] = np.mean(category_values) * 10  # 转换为0-10分

        return scores

    def _generate_recommendation(self, prob: float,
                                  scores: Dict[str, float]) -> str:
        """生成投资建议"""
        if prob >= 0.7:
            return "全力投入：高成功概率，建议大规模资源投入"
        elif prob >= 0.5:
            return "试点测试：中等潜力，建议小批量试水"
        elif prob >= 0.3:
            return "谨慎观望：存在风险，建议进一步优化后再评估"
        else:
            return "暂缓观望：成功概率较低，建议重新考虑产品定位或放弃"

    def _assess_risk_level(self, prob: float) -> str:
        """评估风险等级"""
        if prob >= 0.7:
            return "低风险"
        elif prob >= 0.5:
            return "中等风险"
        elif prob >= 0.3:
            return "高风险"
        else:
            return "极高风险"

    def batch_predict(self, products: List[Dict[str, str]]) -> List[Dict]:
        """
        批量预测多个新品

        Args:
            products: 多个产品的18维度评估列表

        Returns:
            预测结果列表
        """
        results = []
        for product in products:
            result = self.predict(product)
            results.append(result)

        # 按成功概率排序
        results.sort(key=lambda x: x['success_probability'], reverse=True)
        return results


# ==================== 示例代码 ====================

def generate_sample_data(n_products: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """生成模拟的新品数据"""
    np.random.seed(42)

    # 生成18维度的特征
    X = np.random.choice([0.0, 0.5, 1.0], size=(n_products, 18))

    # 生成标签（基于加权规则）
    # 市场维度权重更高
    weights = np.array([
        0.08, 0.08, 0.07, 0.05,  # market
        0.07, 0.08, 0.05, 0.05,  # product
        0.05, 0.06, 0.05, 0.05,  # execution
        0.08, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04  # timing
    ])

    # 计算成功分数
    scores = X @ weights
    # 添加噪声
    scores += np.random.normal(0, 0.1, n_products)

    # 转换为二分类标签
    y = (scores > scores.mean()).astype(int)

    return X, y


def demo_single_product():
    """单个产品预测演示"""
    print("=" * 60)
    print("新品机会挖掘模型 - 单个产品预测")
    print("=" * 60)

    # 1. 初始化模型
    miner = ProductOpportunityMiner(use_llm=False)

    # 2. 生成训练数据
    print("\n[1] 生成训练数据...")
    X, y = generate_sample_data(n_products=1000)
    print(f"   样本数: {len(X)}")
    print(f"   成功率: {y.mean()*100:.1f}%")

    # 3. 训练模型
    print("\n[2] 训练模型...")
    miner.fit(X, y)

    # 4. 模拟新品评估
    print("\n[3] 模拟新品评估...")
    product_desc = "智能穿戴式吸奶器，可连接APP追踪奶量，静音设计，适合职场妈妈"
    market_desc = "美国母婴市场，穿戴式吸奶器需求快速增长，竞品较少，价格敏感度高"

    dimensions = miner.extract_dimensions_with_llm(product_desc, market_desc)

    # 5. 预测
    print("\n[4] 预测结果...")
    result = miner.predict(dimensions)

    print(f"\n产品: {product_desc[:30]}...")
    print(f"成功概率: {result['success_probability']*100:.1f}%")
    print(f"预测结果: {result['prediction']}")
    print(f"风险等级: {result['risk_level']}")
    print(f"\n维度得分:")
    for category, score in result['category_scores'].items():
        print(f"  {category}: {score:.2f}/10")
    print(f"\n投资建议: {result['recommendation']}")

    return miner, result


def demo_batch_comparison():
    """批量产品对比演示"""
    print("\n" + "=" * 60)
    print("新品机会挖掘模型 - 批量产品对比")
    print("=" * 60)

    # 初始化并训练
    miner = ProductOpportunityMiner(use_llm=False)
    X, y = generate_sample_data(n_products=1000)
    miner.fit(X, y)

    # 模拟3个不同产品
    products = [
        {
            'market_size': 'Large', 'growth_rate': 'High', 'industry_trend': 'Rising',
            'regulatory_env': 'Moderate', 'innovation_level': 'High', 'product_market_fit': 'Strong',
            'scalability': 'High', 'user_engagement': 'High', 'development_speed': 'Medium',
            'team_expertise': 'High', 'funding_status': 'Sufficient', 'supply_chain': 'Stable',
            'market_timing': 'Just_Right', 'competition_intensity': 'Medium', 'seasonality': 'Favorable',
            'platform_policy': 'Supportive', 'ip_protection': 'Strong', 'investor_backing': 'Strong'
        },  # 产品A：智能穿戴式吸奶器
        {
            'market_size': 'Medium', 'growth_rate': 'Medium', 'industry_trend': 'Stable',
            'regulatory_env': 'Easy', 'innovation_level': 'Medium', 'product_market_fit': 'Moderate',
            'scalability': 'Medium', 'user_engagement': 'Medium', 'development_speed': 'Fast',
            'team_expertise': 'Medium', 'funding_status': 'Adequate', 'supply_chain': 'Stable',
            'market_timing': 'Just_Right', 'competition_intensity': 'High', 'seasonality': 'Neutral',
            'platform_policy': 'Neutral', 'ip_protection': 'Weak', 'investor_backing': 'Moderate'
        },  # 产品B：普通辅食机配件
        {
            'market_size': 'Small', 'growth_rate': 'Low', 'industry_trend': 'Declining',
            'regulatory_env': 'Difficult', 'innovation_level': 'Low', 'product_market_fit': 'Weak',
            'scalability': 'Low', 'user_engagement': 'Low', 'development_speed': 'Slow',
            'team_expertise': 'Low', 'funding_status': 'Limited', 'supply_chain': 'Unstable',
            'market_timing': 'Late', 'competition_intensity': 'High', 'seasonality': 'Unfavorable',
            'platform_policy': 'Restrictive', 'ip_protection': 'Weak', 'investor_backing': 'Weak'
        },  # 产品C：过时产品
    ]

    product_names = ['智能穿戴式吸奶器', '普通辅食机配件', '过时品类']

    print("\n[批量预测结果排名]")
    results = miner.batch_predict(products)

    for i, (result, name) in enumerate(zip(results, product_names), 1):
        print(f"\n{i}. {name}")
        print(f"   成功概率: {result['success_probability']*100:.1f}%")
        print(f"   投资建议: {result['recommendation']}")
        print(f"   综合得分: {sum(result['category_scores'].values())/4:.2f}/10")


def main():
    """主函数"""
    # 单个产品预测
    miner, result = demo_single_product()

    # 批量对比
    demo_batch_comparison()

    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    print("\n说明：")
    print("1. 实际应用时需要使用真实的历史新品数据进行训练")
    print("2. LLM维度提取需要配置OpenAI API密钥")
    print("3. 18维度评估需要人工输入或对接数据源")
    print("4. 建议每季度更新模型以保持预测准确性")

    return miner


if __name__ == '__main__':
    miner = main()
```

---

## ④ 技能关系

### 前置依赖技能
- **用户流失预测** - 理解预测模型的基本方法论
- **A/B测试** - 验证新品预测准确性的实验设计
- **因果推断** - 理解多维特征对产品成功的影响

### 可组合技能
- **冷启动推荐** - 新品上市后的用户推荐策略
- **需求预测** - 结合成功预测进行销量预估
- **购物篮分析** - 发现新品与现有产品的搭配机会

### 后续扩展技能
- **动态定价** - 基于成功预测调整新品定价策略
- **库存优化** - 根据预测结果优化首批库存
- **供应链规划** - 成功预测指导供应链资源配置

---

## ⑤ 业务价值评估

| 维度 | 评估 |
|------|------|
| **ROI估算** | 假设年新品投入500万，准确率77%可优化决策质量，预计节省无效投入30-50%（150-250万/年），提升整体ROI 15-25% |
| **实施难度** | ★★★☆☆ (3/5) - 需要历史新品数据积累，18维度评估需要人工输入或对接多数据源 |
| **数据需求** | ★★★★☆ (4/5) - 需要新品历史表现数据、市场数据、产品特征数据 |
| **模型复杂度** | ★★★☆☆ (3/5) - Random Forest易于解释，LLM特征提取增加复杂度 |
| **可解释性** | ★★★★☆ (4/5) - 18维度透明评估，维度重要性可量化 |
| **优先级评分** | 85/100 - 高价值决策支持工具，适合新品密集上新的业务场景 |

### 适用场景
- ✅ 月均上新10+ SKU的快速消费品品类
- ✅ 需要决策资源分配的新品开发团队
- ✅ 希望降低新品失败率的成熟卖家
- ✅ 多平台布局需要选择首发平台的场景

### 不适用场景
- ❌ 极度创新的全新品类（缺乏可比历史数据）
- ❌ 季节性极强的短期爆款（时机维度占主导）
- ❌ 数据积累不足的新业务线

---

## 参考论文

- **SSFF: Investigating LLM Predictive Capabilities for Startup Success through a Multi-Agent Framework with Enhanced Explainability and Performance**, arXiv:2405.19456, 2024
- 框架创新：LLM + Random Forest + 18维度评估 + Founder-Idea Fit Network
- 核心指标：77%预测准确率（GPT-4o + 1400样本）
