# Skill: Kano需求分类与优先级 - 从VOC到产品路线图

## 基础信息

- **arXiv ID**: 2303.03798 (Kano分类) + 2603.28677 (iReFeed优先级)
- **论文标题**: 
  - Automatically Classifying Kano Model Factors in App Reviews
  - Enhancing User-Feedback Driven Requirements Prioritization (iReFeed)
- **发表会议/期刊**: 
  - arXiv 2023 (Kano)
  - arXiv 2026 (iReFeed)
- **核心方法**: BERT监督分类 + BERTopic主题聚类 + NSGA-II优化

---

## 1. 算法原理

### 1.1 Kano模型基础

Kano模型将产品功能分为5个类别，指导资源分配决策：

| 类别 | 英文 | 特点 | 产品策略 |
|------|------|------|---------|
| **基本型需求** | Must-be/Basic | 存在不会满意，缺失会极度不满 | 必须满足，优先保障 |
| **绩效型需求** | Performance | 实现程度与满意度线性相关 | 投入产出比优化 |
| **魅力型需求** | Delighters/Attractive |  unexpected会产生惊喜，缺失不会不满 | 差异化竞争重点 |
| **无关型需求** | Indifferent | 有无都不影响满意度 | 避免浪费资源 |
| **排斥型需求** | Reversal | 存在反而导致不满 | 识别并避免 |

```
满意度
  ↑
  │      魅力型       绩效型
  │         ↗       ↗
  │       ↗       ↗
  │     ↗       ↗
  │   ↗       ↗
  │ ↗       ↗
  └────────────────────────→ 功能实现程度
  │       ↘
  │         ↘  基本型
  │           ↘
  │             ↘
  └────────────────────────
  不满
```

### 1.2 Kano自动分类算法

**传统方法的问题**：
- 依赖Kano问卷调研，成本高、样本少
- 基于情感分析的间接分类准确率只有30-63% F1

**BERT直接分类方法**：
```python
# 将需求分类转化为监督学习问题
输入: 用户评论文本 (如"吸力很强"、"噪音太大了")
输出: Kano类别标签

模型: BERT/RoBERTa fine-tuning
训练数据: 8,126条人工标注评论
准确率: 92.8% (远超传统方法)
```

**反直觉洞察**：
直接训练BERT分类器（92.8%准确率）比模拟Kano问卷的情感分析路径（30-63% F1）效果提升50%以上。

### 1.3 iReFeed需求优先级排序

**Next Release Problem (NRP)**：
- 在预算/时间约束下，选择最优需求子集
- 传统方法将需求视为独立实体
- 忽略了需求间的依赖关系

**iReFeed创新**：
```
输入: 用户反馈评论 + 候选需求列表
输出: 最优需求发布序列

流程:
1. BERTopic主题聚类
   - 将用户反馈聚类为主题簇
   - 识别"静音"、"便携"、"续航"等主题

2. 需求关联挖掘
   - 使用ChatGPT自动生成"requires"关系
   - 例如: "快速充电" requires "Type-C接口"

3. NSGA-II多目标优化
   - 目标1: 最大化用户价值
   - 目标2: 最小化开发成本
   - 目标3: 最大化主题一致性
   - 约束: 满足"requires"依赖关系
```

**反直觉洞察**：
考虑需求间关联性的优先级排序比独立评估提升F1-score达35%，因为真实产品的功能是相互依赖的。

---

## 2. 业务应用

### 2.1 Momcozy场景：吸奶器功能Kano分类与路线图

**场景背景**：
- 已有：TopicImpact提取的吸奶器功能主题（吸力、噪音、便携等）
- 问题：不知道哪些功能必须做、哪些可以做差异化
- 目标：生成科学的产品迭代路线图

**应用流程**：

```python
# 步骤1: 从VOC提取功能需求
features = [
    "静音设计", "快速充电", "便携包", "智能APP",
    "记忆模式", "防回流", "LED夜灯", "多档位调节"
]

# 步骤2: Kano自动分类
kano_classifier = KanoClassifier()
for feature in features:
    # 获取该功能相关评论
    reviews = get_feature_reviews(feature)
    # BERT分类
    category = kano_classifier.classify(reviews)
    print(f"{feature}: {category}")

# 输出示例:
# 静音设计: 基本型 (用户认为吸奶器本该静音，不静音会极度不满)
# 智能APP: 魅力型 (有会惊喜，没有也能接受)
# 便携包: 绩效型 (越好越满意)
# LED夜灯: 无关型 (大部分用户不在意)

# 步骤3: 识别功能依赖
requires_graph = {
    "快速充电": ["Type-C接口"],
    "智能APP": ["蓝牙连接", "数据传输"],
    "记忆模式": ["多档位调节"]
}

# 步骤4: NSGA-II生成路线图
roadmap = irefeed.generate_roadmap(
    features=features,
    kano_categories=kano_categories,
    requires_graph=requires_graph,
    budget_constraints={"Q1": 100, "Q2": 150, "Q3": 120}
)
```

### 2.2 产品决策矩阵

```python
decision_matrix = {
    "静音设计": {
        "kano": "基本型",
        "priority": "P0",  # 最高优先级
        "strategy": "必须满足，投入必要资源保障",
        "risk": "如果不做，会导致大量用户流失"
    },
    "吸力调节": {
        "kano": "绩效型",
        "priority": "P1",
        "strategy": "持续优化，与竞品对标",
        "roi": "投入产出比高"
    },
    "智能APP": {
        "kano": "魅力型",
        "priority": "P2",
        "strategy": "差异化亮点，面向高端用户",
        "timing": "Q3发布，配合新品上市"
    },
    "彩色外壳": {
        "kano": "无关型",
        "priority": "Won't Do",
        "strategy": "暂缓，避免资源浪费"
    }
}
```

### 2.3 与竞品的功能对标

```python
# 竞品功能Kano分类对比
competitive_analysis = {
    "Momcozy": {
        "静音": "基本型 ✓",
        "APP": "魅力型 (开发中)",
        "便携": "绩效型 ✓"
    },
    "Medela": {
        "静音": "基本型 ✓",
        "APP": "魅力型 ✓",
        "便携": "绩效型 ✓"
    },
    "Spectra": {
        "静音": "基本型 ✓",
        "APP": "无关型 (没有)",
        "便携": "绩效型 ✗"
    }
}

# 策略建议：
# - 静音是基本型，必须对标Medela
# - APP是魅力型，可以差异化竞争
# - Spectra便携弱，我们可以强化便携作为卖点
```

---

## 3. 代码模板

完整代码见：`paper2skills-code/nlp_voc/kano_requirements_prioritization/model.py`

```python
"""
Kano需求分类与iReFeed优先级排序
从VOC到产品路线图 - Momcozy场景

论文来源:
- Automatically Classifying Kano Model Factors in App Reviews (arXiv:2303.03798)
- Enhancing User-Feedback Driven Requirements Prioritization (arXiv:2603.28677)
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict


class KanoCategory(Enum):
    """Kano模型分类"""
    MUST_BE = "基本型"      # 必须具备
    PERFORMANCE = "绩效型"   # 越多越好
    DELIGHTER = "魅力型"     # 惊喜特性
    INDIFFERENT = "无关型"   # 无所谓
    REVERSAL = "排斥型"      # 越多越差


@dataclass
class Feature:
    """产品功能"""
    name: str
    description: str
    related_reviews: List[str]
    estimated_cost: float  # 开发成本
    estimated_value: float  # 预期用户价值


@dataclass
class KanoClassification:
    """Kano分类结果"""
    feature: Feature
    category: KanoCategory
    confidence: float
    evidence: List[str]  # 支撑评论


class KanoClassifier:
    """
    Kano自动分类器

    使用BERT fine-tuning对用户评论进行Kano分类
    """

    def __init__(self):
        # 关键词规则（模拟BERT输出，实际应使用fine-tuned模型）
        self.keywords = {
            KanoCategory.MUST_BE: [
                "必须", "应该", "本该", "基本", "必备",
                "没有不行", "不可或缺", "底线"
            ],
            KanoCategory.PERFORMANCE: [
                "更好", "更强", "更快", "更静音", "提升",
                "改进", "优化", "希望", "期待"
            ],
            KanoCategory.DELIGHTER: [
                "惊喜", "没想到", "贴心", "智能", "人性化",
                "超预期", "还有这种功能", "赞"
            ],
            KanoCategory.INDIFFERENT: [
                "无所谓", "用不上", "不需要", "多余",
                "鸡肋", "噱头"
            ]
        }

    def classify(self, reviews: List[str]) -> KanoClassification:
        """
        对功能相关评论进行Kano分类

        Args:
            reviews: 功能相关用户评论

        Returns:
            KanoClassification: 分类结果
        """
        # 合并所有评论
        all_text = ' '.join(reviews)

        # 计算各分类得分（模拟BERT预测）
        scores = {}
        for category, keywords in self.keywords.items():
            score = sum(1 for kw in keywords if kw in all_text)
            scores[category] = score

        # 选择最高分类
        if max(scores.values()) == 0:
            best_category = KanoCategory.PERFORMANCE  # 默认绩效型
            confidence = 0.5
        else:
            best_category = max(scores, key=scores.get)
            total = sum(scores.values())
            confidence = scores[best_category] / total if total > 0 else 0.5

        # 提取支撑证据
        evidence = []
        for review in reviews[:3]:  # 取前3条
            evidence.append(review)

        return KanoClassification(
            feature=None,  # 外部设置
            category=best_category,
            confidence=confidence,
            evidence=evidence
        )

    def batch_classify(self, features: List[Feature]) -> List[KanoClassification]:
        """批量分类"""
        results = []
        for feature in features:
            classification = self.classify(feature.related_reviews)
            classification.feature = feature
            results.append(classification)
        return results


class RequirementDependencyGraph:
    """需求依赖关系图"""

    def __init__(self):
        self.requires = defaultdict(set)  # A requires B
        self.incompatibles = defaultdict(set)  # A incompatible with B

    def add_requires(self, feature_a: str, feature_b: str):
        """添加依赖关系: A requires B"""
        self.requires[feature_a].add(feature_b)

    def get_dependencies(self, feature: str) -> Set[str]:
        """获取功能的所有依赖（递归）"""
        deps = set()
        to_process = [feature]
        while to_process:
            current = to_process.pop()
            for dep in self.requires[current]:
                if dep not in deps:
                    deps.add(dep)
                    to_process.append(dep)
        return deps

    def is_valid_selection(self, selected: Set[str]) -> bool:
        """检查选择是否满足所有依赖"""
        for feature in selected:
            deps = self.requires[feature]
            if not deps.issubset(selected):
                return False
        return True


class NSGAIIPrioritizer:
    """
    NSGA-II多目标优化优先级排序

    解决Next Release Problem
    """

    def __init__(
        self,
        population_size: int = 100,
        generations: int = 50,
        mutation_rate: float = 0.1
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def optimize(
        self,
        features: List[Feature],
        kano_results: List[KanoClassification],
        dependency_graph: RequirementDependencyGraph,
        budget: float
    ) -> List[Tuple[Feature, float]]:
        """
        NSGA-II优化需求选择

        Args:
            features: 候选功能列表
            kano_results: Kano分类结果
            dependency_graph: 依赖关系图
            budget: 预算约束

        Returns:
            List[(Feature, priority_score)]: 优先级排序结果
        """
        n = len(features)

        # 目标函数1: 最大化用户价值
        def objective_value(selection: np.ndarray) -> float:
            value = 0.0
            for i, selected in enumerate(selection):
                if selected:
                    # Kano权重调整
                    kano = kano_results[i].category
                    multiplier = {
                        KanoCategory.MUST_BE: 3.0,
                        KanoCategory.PERFORMANCE: 2.0,
                        KanoCategory.DELIGHTER: 1.5,
                        KanoCategory.INDIFFERENT: 0.5,
                        KanoCategory.REVERSAL: -1.0
                    }.get(kano, 1.0)
                    value += features[i].estimated_value * multiplier
            return -value  # 最小化负值 = 最大化正值

        # 目标函数2: 最小化成本
        def objective_cost(selection: np.ndarray) -> float:
            cost = sum(
                features[i].estimated_cost
                for i, selected in enumerate(selection)
                if selected
            )
            return cost

        # 约束检查
        def is_valid(selection: np.ndarray) -> bool:
            selected_features = {
                features[i].name
                for i, selected in enumerate(selection)
                if selected
            }
            return dependency_graph.is_valid_selection(selected_features)

        # 简化版贪心算法（实际应使用完整NSGA-II）
        return self._greedy_optimize(
            features, kano_results, dependency_graph, budget
        )

    def _greedy_optimize(
        self,
        features: List[Feature],
        kano_results: List[KanoClassification],
        dependency_graph: RequirementDependencyGraph,
        budget: float
    ) -> List[Tuple[Feature, float]]:
        """贪心算法简化版"""
        # 计算价值密度 = 价值 / 成本
        value_density = []
        for i, (feature, kano) in enumerate(zip(features, kano_results)):
            kano_multiplier = {
                KanoCategory.MUST_BE: 3.0,
                KanoCategory.PERFORMANCE: 2.0,
                KanoCategory.DELIGHTER: 1.5,
                KanoCategory.INDIFFERENT: 0.5,
                KanoCategory.REVERSAL: -1.0
            }.get(kano.category, 1.0)

            adjusted_value = feature.estimated_value * kano_multiplier * kano.confidence
            density = adjusted_value / (feature.estimated_cost + 1)
            value_density.append((i, density, adjusted_value))

        # 按价值密度排序
        value_density.sort(key=lambda x: x[1], reverse=True)

        # 贪心选择（满足依赖约束）
        selected = set()
        total_cost = 0.0
        results = []

        for idx, density, value in value_density:
            feature = features[idx]

            # 检查预算
            if total_cost + feature.estimated_cost > budget:
                continue

            # 检查依赖
            deps = dependency_graph.get_dependencies(feature.name)
            if not deps.issubset({features[i].name for i in selected}):
                continue

            selected.add(idx)
            total_cost += feature.estimated_cost
            results.append((feature, value))

        return results


class iReFeedPrioritizer:
    """
    iReFeed需求优先级排序器

    整合BERTopic主题聚类、依赖挖掘和NSGA-II优化
    """

    def __init__(self):
        self.kano_classifier = KanoClassifier()
        self.dependency_graph = RequirementDependencyGraph()
        self.optimizer = NSGAIIPrioritizer()

    def analyze_and_prioritize(
        self,
        features: List[Feature],
        budget: float,
        quarter_budgets: Dict[str, float] = None
    ) -> Dict:
        """
        完整分析和优先级排序流程

        Args:
            features: 候选功能列表
            budget: 总预算
            quarter_budgets: 季度预算分配

        Returns:
            Dict: 完整分析报告
        """
        # 步骤1: Kano分类
        print("【步骤1】Kano自动分类...")
        kano_results = self.kano_classifier.batch_classify(features)

        for result in kano_results:
            print(f"  {result.feature.name}: {result.category.value} "
                  f"(置信度: {result.confidence:.2f})")

        # 步骤2: 挖掘依赖关系（模拟ChatGPT输出）
        print("\n【步骤2】挖掘需求依赖关系...")
        self._auto_detect_dependencies(features)

        # 步骤3: NSGA-II优化
        print("\n【步骤3】NSGA-II多目标优化...")
        prioritized = self.optimizer.optimize(
            features, kano_results, self.dependency_graph, budget
        )

        # 步骤4: 生成路线图
        print("\n【步骤4】生成产品路线图...")
        roadmap = self._generate_roadmap(
            prioritized, kano_results, quarter_budgets
        )

        return {
            'kano_classifications': kano_results,
            'dependencies': dict(self.dependency_graph.requires),
            'prioritized_features': prioritized,
            'roadmap': roadmap
        }

    def _auto_detect_dependencies(self, features: List[Feature]):
        """自动检测功能依赖（模拟LLM输出）"""
        # 基于功能名称的规则推断（实际应使用LLM）
        feature_names = {f.name: f for f in features}

        # 常见依赖模式
        if "智能APP" in feature_names:
            if "蓝牙连接" in feature_names:
                self.dependency_graph.add_requires("智能APP", "蓝牙连接")
            if "数据传输" in feature_names:
                self.dependency_graph.add_requires("智能APP", "数据传输")

        if "快速充电" in feature_names and "Type-C接口" in feature_names:
            self.dependency_graph.add_requires("快速充电", "Type-C接口")

        if "记忆模式" in feature_names and "多档位调节" in feature_names:
            self.dependency_graph.add_requires("记忆模式", "多档位调节")

    def _generate_roadmap(
        self,
        prioritized: List[Tuple[Feature, float]],
        kano_results: List[KanoClassification],
        quarter_budgets: Dict[str, float]
    ) -> Dict[str, List[Dict]]:
        """生成季度路线图"""
        if quarter_budgets is None:
            quarter_budgets = {"Q1": float('inf')}

        roadmap = {q: [] for q in quarter_budgets.keys()}
        kano_map = {r.feature.name: r for r in kano_results}

        quarter_list = sorted(quarter_budgets.keys())
        current_quarter = 0
        remaining_budget = quarter_budgets[quarter_list[0]]

        for feature, priority in prioritized:
            # 找到当前季度
            while current_quarter < len(quarter_list):
                q = quarter_list[current_quarter]
                if feature.estimated_cost <= remaining_budget:
                    kano = kano_map[feature.name]
                    roadmap[q].append({
                        'feature': feature.name,
                        'priority_score': priority,
                        'kano': kano.category.value,
                        'cost': feature.estimated_cost,
                        'strategy': self._get_strategy(kano.category)
                    })
                    remaining_budget -= feature.estimated_cost
                    break
                else:
                    current_quarter += 1
                    if current_quarter < len(quarter_list):
                        remaining_budget = quarter_budgets[quarter_list[current_quarter]]

        return roadmap

    def _get_strategy(self, category: KanoCategory) -> str:
        """根据Kano类别返回策略建议"""
        strategies = {
            KanoCategory.MUST_BE: "P0-必须满足，投入必要资源",
            KanoCategory.PERFORMANCE: "P1-持续优化，对标竞品",
            KanoCategory.DELIGHTER: "P2-差异化亮点，惊喜用户",
            KanoCategory.INDIFFERENT: "暂不投入，观察反馈",
            KanoCategory.REVERSAL: "避免实现"
        }
        return strategies.get(category, "待定")


# ==================== Momcozy业务场景示例 ====================

def generate_momcozy_features() -> List[Feature]:
    """生成Momcozy功能需求示例"""
    features = [
        Feature(
            name="静音设计",
            description="降低吸奶器工作噪音",
            related_reviews=[
                "噪音太大了，在公司用很尴尬",
                "必须静音，不然影响宝宝睡觉",
                "静音模式是底线要求"
            ],
            estimated_cost=80,
            estimated_value=90
        ),
        Feature(
            name="智能APP",
            description="手机APP控制和记录",
            related_reviews=[
                "没想到还有APP记录功能，太贴心了",
                "智能功能挺惊喜的",
                "年轻人喜欢科技感"
            ],
            estimated_cost=120,
            estimated_value=60
        ),
        Feature(
            name="快速充电",
            description="支持快充技术",
            related_reviews=[
                "充电越快越好",
                "希望充电速度能提升",
                "快充很实用"
            ],
            estimated_cost=40,
            estimated_value=70
        ),
        Feature(
            name="Type-C接口",
            description="通用充电接口",
            related_reviews=[
                "Type-C很方便",
                "不用多带线了"
            ],
            estimated_cost=15,
            estimated_value=50
        ),
        Feature(
            name="彩色外壳",
            description="多种颜色可选",
            related_reviews=[
                "颜色无所谓，好用就行",
                "不太在意颜色"
            ],
            estimated_cost=30,
            estimated_value=20
        ),
        Feature(
            name="记忆模式",
            description="记住常用设置",
            related_reviews=[
                "每次都要调很麻烦",
                "记忆功能不错"
            ],
            estimated_cost=25,
            estimated_value=55
        ),
        Feature(
            name="多档位调节",
            description="吸力多档可调",
            related_reviews=[
                "档位越多越好",
                "希望能更精细调节"
            ],
            estimated_cost=20,
            estimated_value=65
        ),
    ]
    return features


def demo():
    """iReFeed完整演示"""
    print("=" * 70)
    print("Kano需求分类与iReFeed优先级排序 - Momcozy吸奶器场景")
    print("=" * 70)

    # 1. 准备功能需求
    print("\n【准备】定义候选功能需求...")
    features = generate_momcozy_features()
    for f in features:
        print(f"  - {f.name} (成本: {f.estimated_cost}, 价值: {f.estimated_value})")

    # 2. 初始化iReFeed
    print("\n【初始化】iReFeed优先级排序器...")
    irefeed = iReFeedPrioritizer()

    # 3. 执行分析和排序
    print("\n【执行】Kano分类 + 依赖挖掘 + NSGA-II优化...")
    result = irefeed.analyze_and_prioritize(
        features=features,
        budget=300,
        quarter_budgets={"Q1": 100, "Q2": 120, "Q3": 80}
    )

    # 4. 展示依赖关系
    print("\n" + "=" * 70)
    print("需求依赖关系")
    print("=" * 70)
    for feature, deps in result['dependencies'].items():
        print(f"  {feature} requires: {deps}")

    # 5. 展示路线图
    print("\n" + "=" * 70)
    print("产品路线图")
    print("=" * 70)
    for quarter, items in result['roadmap'].items():
        print(f"\n【{quarter}】")
        for item in items:
            print(f"  - {item['feature']} ({item['kano']})")
            print(f"    策略: {item['strategy']}")

    # 6. 业务价值总结
    print("\n" + "=" * 70)
    print("业务价值")
    print("=" * 70)
    print("✓ TopicImpact主题 → Kano分类 → 优先级排序")
    print("✓ 识别基本型需求（必须保障）vs 魅力型需求（差异化）")
    print("✓ 考虑功能依赖关系，生成可行路线图")
    print("✓ 资源分配决策从主观判断转向数据驱动")


if __name__ == '__main__':
    demo()
