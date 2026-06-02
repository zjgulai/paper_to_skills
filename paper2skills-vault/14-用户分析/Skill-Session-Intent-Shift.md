---
title: Session意图漂移建模 - 跨会话用户购买意图变化检测
doc_type: knowledge
module: 14-用户分析
topic: session-intent-shift
status: stable
created: 2026-05-20
updated: 2026-05-20
owner: self
source: human+ai
paper: arXiv:2507.20185 (2025)
---

# Skill: Session Intent Shift Modeling — 跨会话用户意图漂移检测

> 论文：**SessionIntentBench: A Multi-task Inter-session Intention-shift Modeling Benchmark for E-commerce Customer Behavior Understanding** · arXiv:2507.20185 (2025)
> 作者：Yuqi Yang, Weiqi Wang 等（HKUST + Amazon），发表于 ACL 2026 Findings
> 应用：Amazon M2数据集的跨Session意图变化追踪，1,952,177意图条目、1,132,145条session意图轨迹

---

## ① 算法原理

### 核心思想

现有电商推荐系统多依赖商品标题、价格等**表层属性**推断用户意图，且只关注**单次购买**或**单会话**内的短期偏好变化。SessionIntentBench 的核心创新在于：提出**意图树（Intention Tree）**概念，通过**跨会话**建模用户意图的时序演化，构建大规模多模态意图基准。

**四步构建流程：**
1. **多模态属性提取**：使用 GPT-4o-mini 从商品文本描述+图片中提取标准化属性-值对（如 `color: white`、`size: 7.5 inches`），用于后续意图推断的证据基础
2. **LLM多步意图推断**：在每个时间步，将前步意图 `{I_i}_{i=1}^{t-1}` 和当前商品注入提示词，推断用户最可能的5个意图分支；从第5个商品起收窄为单意图追踪，控制树规模指数爆炸
3. **意图树构建（时间展开）**：每个会话对应一棵意图树，根节点为初始意图，子节点为每步分支意图，时序连接构成意图轨迹
4. **跨会话漂移原因分析**：提取每次意图转变背后的**关键属性** `A_t`（如 `price: cheaper`）和**对比依据** `C_t`（`P_t vs P_{t-1} 的差异`）

**4个评估子任务：**

| 任务 | 输入 | 输出 | 考察能力 |
|------|------|------|---------|
| **Task1** 意图-商品匹配似然 | 历史 `ℋ_{t-1}` + 意图 `I_{t-1}` + 新商品 `P_t` | 似然分 ∈ {0,1,2,3} | 意图与行为对齐 |
| **Task2** 属性驱动购买似然 | 历史 `ℋ_{t-1}` + 关键属性 `A_{t-1}` + 新商品 `P_t` | 似然分 ∈ {0,1,2,3} | 属性正则化意图 |
| **Task3** 意图合理性验证 | 比较依据 `C_t` + 前后商品 `P_{t-1}, P_t` + 意图 `I_{t-1}, I_t` | 合理性分 ∈ {0,1,2,3} | 防止意图幻觉 |
| **Task4** 意图演化预测 | 全部历史 `ℋ_t` + 意图 `I_t` | 下一步探索度 ∈ {1,2,3} | 推荐策略决策 |

### 数学直觉

**会话交互历史定义：**
```
ℋ_t = {(P_j, A_j)}_{j=1}^{t}
```
其中 `P_j` 为第 j 步商品，`A_j` 为该步关键属性。

**意图树分支规则：**
- 前4步：每步输出5个可能意图分支 → 树呈指数增长用于训练集多样性
- 第5步起：单意图追踪（|New Intent|=1）→ 控制 token 消耗

**Task4 探索-利用三态映射：**
```
Score=1: Exploit-Same     → 同类目相似商品（已锁定偏好，深度利用）
Score=2: Exploit-Variant  → 同类目不同特征（局部探索）
Score=3: Explore-New      → 跨类目新商品（全局探索，偏好仍未定型）
```

### 关键假设

1. **意图连续性**：用户在同一 session 内的意图演化存在可追溯的因果链，而非随机跳变
2. **属性可观测性**：意图变化背后的关键驱动属性（价格/颜色/材质）可从商品多模态信息中提取
3. **跨模态一致性**：文本描述与商品图片传递一致的属性信号，LLM 可综合两者进行意图推断
4. **意图注入有效性**：显式意图信息能帮助下游 LLM 完成 session 理解任务（实验验证成立）

### 关键效果数字

**数据集规模（基于 Amazon M2 + Amazon Review Dataset）：**
- 筛选后完整多模态会话：**10,905 个**
- 意图条目总数：**1,952,177 条**
- 意图轨迹总数：**1,132,145 条**
- 可用任务实例：**13,003,664 个**
- 人工标注子集：**8,980 条**（Amazon Mechanical Turk）

**意图注入效果（注入 vs 不注入）：**
| 任务 | 提升幅度 |
|------|---------|
| Task1：意图-商品匹配似然 | **+1.75%** |
| Task2：属性驱动购买似然 | **+3.09%** |
| Task4：意图演化预测 | **+4.24%** |

**模型总体表现**：20+个 L(V)LM（包括 GPT-4o、LLaMA 系列、Claude 系列等）在 4 个子任务上均显著低于人类基准，证明跨会话意图理解仍是开放难题。微调后（SFT on SessionIntentBench）小模型性能大幅提升。

---

## ② 母婴出海应用案例

### 场景1：桑基图路径语义标注

**业务问题**：桑基图展示了页面间流量（如"首页→分类页→PDP→加购→支付"），但缺少"为什么用户走这条路"的语义信息。例如同样是"PDP → 加购"路径，有的用户是**目标型**（直接搜索特定型号奶粉来购买），有的是**探索型**（随机浏览发现心仪产品），有的是**比价型**（连续访问多个 PDP 后才加购）。意图漂移检测可以为桑基图每条路径的"边"标注语义标签，让运营人员直观理解流量结构。

**数据要求**：

| 字段 | 类型 | 示例 |
|------|------|------|
| user_id | string | "usr_abc123" |
| session_id | string | "sess_2026042001" |
| event_type | category | "view" / "click" / "add_cart" / "purchase" |
| page_type | category | "homepage" / "category" / "pdp" / "cart" / "checkout" |
| product_id | string | "ASIN_B08XYZ123" |
| product_title | string | "Similac Pro-Advance Infant Formula..." |
| product_category | string | "Baby Formula" |
| product_price | float | 45.99 |
| event_timestamp | datetime | "2026-04-20 14:32:05" |

最低要求：至少3个连续商品访问构成一个完整会话，有可提取的商品属性信息。

**预期产出**：为桑基图路径标注意图标签（探索/比较/目标购买/复购），并输出每段转跳的意图漂移原因
**业务价值**：帮助运营识别"高价值路径"（目标购买型）vs"低效路径"（探索后流失），优化页面布局和推荐位排布，预计提升关键路径转化 **10-15%**

### 场景2：孕期→产后意图漂移的营销时机

**业务问题**：母婴用户从孕期搜索"孕妇装"到产后搜索"吸奶器""纸尿裤"，意图随时间自然漂移。这种跨会话（跨天/跨周）的意图转变信号往往在用户明确表达需求之前就已可观测——当用户的 Task4 评分从"利用"转向"探索"时，正是营销介入的最佳时机窗口。

**具体表现**：
- 孕中期：PDP 停留时间长、多次比较婴儿车 → **比较型意图**（Task4 Score=2）
- 孕晚期：搜索关键词变化（"新生儿用品""待产包"）→ **意图漂移检测点**（Task2 属性突变）
- 产后：直接搜索"吸奶器型号"、"纸尿裤 NB 码" → **目标购买型意图**（Task4 Score=1）

检测到意图漂移的时间窗口（Stage2→Stage3 的转变），精准触达：推送"产后必备清单"而非继续推荐孕期商品，避免营销资源浪费，预计提升相关 EDM/Push 点击率 **20-30%**。

---

## ③ 代码模板

```python
"""
Session Intent Shift Detection — 跨会话意图漂移检测
arXiv: 2507.20185 | SessionIntentBench (ACL 2026 Findings)

功能：
  1. 从会话数据构建意图树（规则+embedding模拟LLM意图推断）
  2. 跨会话意图漂移检测（余弦相似度 + 属性变化追踪）
  3. 为桑基图路径标注意图语义标签
  4. 意图演化阶段预测（探索/比较/目标购买）

环境依赖: pip install numpy scikit-learn
可选依赖: pip install openai  # 如需接入真实LLM意图推断
"""

import math
import random
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# 1. 数据结构
# ─────────────────────────────────────────────

@dataclass
class Product:
    """电商商品（母婴品类）"""
    product_id: str
    title: str
    category: str          # 品类：如 "Baby Formula", "Stroller", "Diaper"
    price: float
    attributes: Dict[str, str] = field(default_factory=dict)
    # 例：{"material": "organic", "age_range": "0-6m", "brand": "Similac"}

    def get_embedding(self, dim: int = 32) -> np.ndarray:
        """
        模拟商品向量表示（真实场景可替换为 text-embedding-ada-002 等）
        用商品属性的哈希值生成确定性伪随机向量
        """
        seed_str = f"{self.category}|{self.price:.0f}|" + "|".join(
            f"{k}:{v}" for k, v in sorted(self.attributes.items())
        )
        seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(dim).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-9)


@dataclass
class IntentionEntry:
    """单步意图条目（对应论文中 Intention Tree 的一个节点）"""
    step: int               # 在会话中的时间步
    intent_text: str        # 意图描述，如 "寻找有机奶粉，追求性价比"
    key_attribute: str      # 驱动此意图的关键属性，如 "price: affordable"
    comparison: str         # 与上一商品的比较依据（Task3 输入）
    intent_vector: np.ndarray = field(default_factory=lambda: np.zeros(32))

    def drift_score(self, other: "IntentionEntry") -> float:
        """计算两个意图之间的漂移程度 (0=完全一致, 1=完全漂移)"""
        sim = cosine_similarity(
            self.intent_vector.reshape(1, -1),
            other.intent_vector.reshape(1, -1)
        )[0][0]
        return float(1.0 - sim)


@dataclass
class SessionIntentTrajectory:
    """一条完整的会话意图轨迹（意图树上的一条路径）"""
    session_id: str
    user_id: str
    products: List[Product] = field(default_factory=list)
    intentions: List[IntentionEntry] = field(default_factory=list)

    def get_exploration_score(self) -> int:
        """
        Task4: 意图演化预测（探索-利用评分）
        1 = 利用（同类目相似商品，意图稳定）
        2 = 局部探索（同类目不同属性）
        3 = 全局探索（跨类目，意图漂移显著）
        """
        if len(self.products) < 2 or len(self.intentions) < 2:
            return 1

        last_product = self.products[-1]
        prev_products = self.products[:-1]

        # 检查是否跨品类（漂移到完全不同的类目）
        categories = [p.category for p in prev_products]
        if last_product.category not in categories:
            return 3

        # 检查意图漂移程度
        last_intent = self.intentions[-1]
        prev_intent = self.intentions[-2]
        drift = last_intent.drift_score(prev_intent)

        if drift > 0.5:
            return 2  # 同类目但属性偏好明显变化
        else:
            return 1  # 稳定利用，继续深入


# ─────────────────────────────────────────────
# 2. 意图推断引擎（规则+Embedding模拟LLM）
# ─────────────────────────────────────────────

# 母婴品类意图模板库
INTENT_TEMPLATES = {
    "Baby Formula": [
        "寻找适合{age_range}宝宝的奶粉，注重{material}成分",
        "比较不同品牌奶粉，价格预算约${price:.0f}",
        "寻找与当前使用奶粉相似但更实惠的替代品",
        "专门搜索有机/DHA强化配方奶粉",
        "复购熟悉品牌，寻找批量优惠",
    ],
    "Stroller": [
        "探索{age_range}适用的婴儿车，关注轻便性",
        "比较折叠便携型婴儿车，价格约${price:.0f}",
        "寻找多功能婴儿车（可躺可坐）",
        "关注安全认证和耐用性，品牌倾向{brand}",
        "寻找城市通勤场景优化的婴儿车",
    ],
    "Diaper": [
        "寻找{age_range}宝宝尺码的纸尿裤，追求吸水性",
        "比较拉拉裤和贴身纸尿裤，批量囤货",
        "关注透气性和不起疹，适合敏感肌",
        "寻找夜用纸尿裤，防漏侧翻",
        "品牌忠诚复购，寻找促销机会",
    ],
    "default": [
        "浏览母婴商品，探索性购物",
        "比较同类商品，寻找最优选择",
        "有明确需求，目标导向搜索",
    ],
}

ATTRIBUTE_CHANGE_SIGNALS = {
    "price_drop": "发现更低价格的同类商品 → 价格敏感型漂移",
    "category_switch": "从{from_cat}切换到{to_cat} → 需求演化（如孕期→产后）",
    "quality_upgrade": "从普通型升级到高端有机型 → 品质追求漂移",
    "brand_switch": "从{from_brand}切换到{to_brand} → 品牌探索漂移",
    "age_range_shift": "从新生儿商品迁移到大月龄商品 → 宝宝成长驱动漂移",
}


def infer_intention(
    product: Product,
    prev_intention: Optional[IntentionEntry] = None,
    intent_dim: int = 32,
) -> IntentionEntry:
    """
    基于商品属性和前步意图推断当前意图
    （生产环境可替换为 LLM API 调用，如 GPT-4o-mini）

    论文原方案：
      prompt = <TASK-PROMPT>
        <Prev Intent>{prev_intention}
        <Prev Products>{prev_products}
        <New Product>{product}
      输出: <New Intent><Attr><Rationale><Comp>
    """
    # 选择意图模板
    templates = INTENT_TEMPLATES.get(product.category, INTENT_TEMPLATES["default"])
    template = random.choice(templates)

    # 填充模板变量
    format_vars = {
        "age_range": product.attributes.get("age_range", "0-12m"),
        "material": product.attributes.get("material", "标准配方"),
        "brand": product.attributes.get("brand", "知名品牌"),
        "price": product.price,
    }
    intent_text = template.format_map({k: format_vars.get(k, "") for k in ["age_range", "material", "brand", "price"]})

    # 提取关键属性（模拟 Task2 的 A_t）
    key_attr_candidates = list(product.attributes.items())
    if product.price < 30:
        key_attr_candidates.append(("price", "affordable"))
    elif product.price > 80:
        key_attr_candidates.append(("price", "premium"))
    key_attribute = ": ".join(key_attr_candidates[0]) if key_attr_candidates else "category: general"

    # 生成比较依据（模拟 Task3 的 C_t）
    if prev_intention is not None:
        comparison = f"当前商品属性({key_attribute})相比前一商品发生变化，驱动意图更新"
    else:
        comparison = "首个商品，建立初始意图基准"

    # 生成意图向量（商品向量 + 部分继承前步意图向量）
    product_vec = product.get_embedding(intent_dim)
    if prev_intention is not None:
        # 意图具有惰性：70%新商品信号 + 30%前步意图
        intent_vec = 0.7 * product_vec + 0.3 * prev_intention.intent_vector
        intent_vec = intent_vec / (np.linalg.norm(intent_vec) + 1e-9)
    else:
        intent_vec = product_vec

    return IntentionEntry(
        step=0,  # 调用方设置
        intent_text=intent_text,
        key_attribute=key_attribute,
        comparison=comparison,
        intent_vector=intent_vec,
    )


# ─────────────────────────────────────────────
# 3. 意图树构建
# ─────────────────────────────────────────────

def build_session_intent_trajectory(
    session_id: str,
    user_id: str,
    products: List[Product],
    n_branches: int = 5,    # 前4步的分支数（论文设定）
    branch_depth: int = 4,  # 开始收窄为单分支的深度
) -> SessionIntentTrajectory:
    """
    构建单个会话的意图轨迹
    实现论文 Section 4.2 的意图树构建逻辑

    前 branch_depth 步：每步生成 n_branches 个候选意图，随机选一个继续
    之后：单意图追踪
    """
    trajectory = SessionIntentTrajectory(
        session_id=session_id,
        user_id=user_id,
        products=products,
    )

    prev_intention = None
    for step, product in enumerate(products):
        if step < branch_depth:
            # 多分支阶段：生成候选意图，选择最高置信度的
            candidates = [
                infer_intention(product, prev_intention)
                for _ in range(n_branches)
            ]
            # 选择与前步意图漂移最小的候选（模拟"最合理"路径）
            if prev_intention is not None:
                scores = [1.0 - c.drift_score(prev_intention) for c in candidates]
                chosen = candidates[np.argmax(scores)]
            else:
                chosen = candidates[0]
        else:
            # 单意图追踪阶段
            chosen = infer_intention(product, prev_intention)

        chosen.step = step
        trajectory.intentions.append(chosen)
        prev_intention = chosen

    return trajectory


# ─────────────────────────────────────────────
# 4. 意图漂移检测与桑基图标注
# ─────────────────────────────────────────────

@dataclass
class SankeyPathLabel:
    """桑基图路径语义标注结果"""
    from_page: str
    to_page: str
    user_count: int
    intent_label: str        # 探索型/比较型/目标购买型/复购型
    drift_score: float       # 0-1，1=完全漂移
    key_attribute: str       # 驱动此转跳的关键属性
    session_ids: List[str] = field(default_factory=list)


def classify_intent_label(
    trajectory: SessionIntentTrajectory,
    step: int,
) -> str:
    """
    为桑基图中的单步转跳标注意图语义

    分类标准（基于 Task4 探索评分 + 漂移程度）：
    - 目标购买型：最近2步意图一致，属于利用阶段
    - 比较型：同品类多次访问，但属性偏好在变化
    - 探索型：跨品类或意图漂移 > 0.5
    - 复购型：品牌/SKU 完全一致的重复访问
    """
    if step == 0 or step >= len(trajectory.intentions):
        return "探索型"

    curr_intent = trajectory.intentions[step]
    prev_intent = trajectory.intentions[step - 1]

    # 检查复购（同品类同品牌）
    if step < len(trajectory.products):
        curr_prod = trajectory.products[step]
        prev_prod = trajectory.products[step - 1]
        curr_brand = curr_prod.attributes.get("brand", "")
        prev_brand = prev_prod.attributes.get("brand", "")
        if curr_brand and curr_brand == prev_brand and curr_prod.category == prev_prod.category:
            return "复购型"

    # 基于意图漂移程度分类
    drift = curr_intent.drift_score(prev_intent)
    exploration = trajectory.get_exploration_score()

    if exploration == 3:
        return "探索型"
    elif drift > 0.35:
        return "比较型"
    else:
        return "目标购买型"


def label_sankey_paths(
    trajectories: List[SessionIntentTrajectory],
    path_pairs: List[Tuple[str, str]],  # [(from_page, to_page), ...]
) -> List[SankeyPathLabel]:
    """
    批量为桑基图路径标注意图语义

    Args:
        trajectories: 所有会话的意图轨迹列表
        path_pairs: 需要标注的页面转跳对列表

    Returns:
        每条路径的语义标注结果
    """
    # 按路径分组（简化：用商品品类代替页面类型）
    path_groups: Dict[Tuple[str, str], List[Tuple[SessionIntentTrajectory, int]]] = {}

    for trajectory in trajectories:
        for step in range(1, len(trajectory.products)):
            from_cat = trajectory.products[step - 1].category
            to_cat = trajectory.products[step].category
            key = (from_cat, to_cat)
            if key not in path_groups:
                path_groups[key] = []
            path_groups[key].append((trajectory, step))

    results = []
    for (from_page, to_page), instances in path_groups.items():
        if not instances:
            continue

        # 统计该路径的意图标签分布
        label_counts: Dict[str, int] = {}
        drift_scores = []
        key_attrs = []
        session_ids = []

        for traj, step in instances:
            label = classify_intent_label(traj, step)
            label_counts[label] = label_counts.get(label, 0) + 1

            curr = traj.intentions[step]
            prev = traj.intentions[step - 1]
            drift_scores.append(curr.drift_score(prev))
            key_attrs.append(curr.key_attribute)
            session_ids.append(traj.session_id)

        # 取最高频意图标签作为该路径的代表标签
        dominant_label = max(label_counts, key=label_counts.get)
        avg_drift = float(np.mean(drift_scores)) if drift_scores else 0.0

        # 最常见的关键属性
        attr_counts: Dict[str, int] = {}
        for attr in key_attrs:
            attr_counts[attr] = attr_counts.get(attr, 0) + 1
        top_attr = max(attr_counts, key=attr_counts.get) if attr_counts else "unknown"

        results.append(SankeyPathLabel(
            from_page=from_page,
            to_page=to_page,
            user_count=len(instances),
            intent_label=dominant_label,
            drift_score=round(avg_drift, 3),
            key_attribute=top_attr,
            session_ids=session_ids[:5],  # 只保存前5个示例
        ))

    return sorted(results, key=lambda x: x.user_count, reverse=True)


# ─────────────────────────────────────────────
# 5. 模拟数据生成
# ─────────────────────────────────────────────

MOTHER_BABY_PRODUCTS = [
    # 配方奶粉
    Product("P001", "Similac Pro-Advance Infant Formula 0-12m", "Baby Formula", 45.99,
            {"brand": "Similac", "material": "organic_DHA", "age_range": "0-12m"}),
    Product("P002", "Enfamil NeuroPro Baby Formula 0-12m", "Baby Formula", 42.50,
            {"brand": "Enfamil", "material": "standard", "age_range": "0-12m"}),
    Product("P003", "Earth's Best Organic Infant Formula", "Baby Formula", 38.99,
            {"brand": "EarthsBest", "material": "organic", "age_range": "0-12m"}),
    Product("P004", "Gerber Good Start Soy Formula 0-12m", "Baby Formula", 35.00,
            {"brand": "Gerber", "material": "soy_based", "age_range": "0-12m"}),
    # 婴儿车
    Product("P005", "UPPAbaby Vista V2 Full-Size Stroller", "Stroller", 999.99,
            {"brand": "UPPAbaby", "weight": "27lb", "age_range": "0-50lb", "foldable": "yes"}),
    Product("P006", "Baby Jogger City Mini GT2 Stroller", "Stroller", 399.99,
            {"brand": "BabyJogger", "weight": "22lb", "age_range": "0-65lb", "foldable": "yes"}),
    Product("P007", "Chicco Bravo Trio Travel System", "Stroller", 329.99,
            {"brand": "Chicco", "weight": "25lb", "age_range": "newborn-50lb", "foldable": "yes"}),
    # 纸尿裤
    Product("P008", "Pampers Swaddlers Newborn Diapers NB", "Diaper", 28.99,
            {"brand": "Pampers", "size": "NB", "count": "84", "feature": "wetness_indicator"}),
    Product("P009", "Huggies Little Snugglers Diapers Size 1", "Diaper", 26.50,
            {"brand": "Huggies", "size": "Size1", "count": "100", "feature": "gentle_skin"}),
    Product("P010", "Honest Company Plant-Based Diapers S2", "Diaper", 35.99,
            {"brand": "Honest", "size": "Size2", "count": "76", "feature": "organic_materials"}),
    # 吸奶器（产后需求）
    Product("P011", "Medela Pump In Style Advanced Breast Pump", "Breast Pump", 159.99,
            {"brand": "Medela", "type": "double_electric", "portable": "yes"}),
    Product("P012", "Spectra S2 Plus Electric Breast Pump", "Breast Pump", 129.99,
            {"brand": "Spectra", "type": "double_electric", "portable": "no"}),
]


def simulate_pregnancy_to_postnatal_sessions(
    n_users: int = 50,
    seed: int = 42,
) -> List[SessionIntentTrajectory]:
    """
    模拟孕期→产后意图漂移的会话数据
    体现 Session 2 场景：意图随时间自然演化
    """
    random.seed(seed)
    np.random.seed(seed)

    trajectories = []
    formula_products = [p for p in MOTHER_BABY_PRODUCTS if p.category == "Baby Formula"]
    stroller_products = [p for p in MOTHER_BABY_PRODUCTS if p.category == "Stroller"]
    diaper_products = [p for p in MOTHER_BABY_PRODUCTS if p.category == "Diaper"]
    pump_products = [p for p in MOTHER_BABY_PRODUCTS if p.category == "Breast Pump"]

    for uid in range(n_users):
        user_id = f"user_{uid:04d}"

        # 阶段1：孕期会话（婴儿车主导）
        if random.random() < 0.7:
            products = random.choices(stroller_products, k=random.randint(2, 4))
            # 有时附带奶粉调研
            if random.random() < 0.4:
                products += random.choices(formula_products, k=1)
            traj = build_session_intent_trajectory(
                session_id=f"{user_id}_pregnancy",
                user_id=user_id,
                products=products,
            )
            trajectories.append(traj)

        # 阶段2：产后会话（奶粉+纸尿裤+吸奶器）
        postnatal_pool = formula_products + diaper_products + pump_products
        products = random.choices(postnatal_pool, k=random.randint(3, 5))
        traj = build_session_intent_trajectory(
            session_id=f"{user_id}_postnatal",
            user_id=user_id,
            products=products,
        )
        trajectories.append(traj)

    return trajectories


# ─────────────────────────────────────────────
# 6. 测试用例
# ─────────────────────────────────────────────

def run_tests():
    """验证核心功能"""
    print("\n" + "=" * 60)
    print("🧪 SessionIntentBench 核心功能测试")
    print("=" * 60)

    # ── Test 1：商品向量生成确定性 ──
    print("\n[Test 1] 商品向量确定性")
    p = MOTHER_BABY_PRODUCTS[0]
    vec1 = p.get_embedding(32)
    vec2 = p.get_embedding(32)
    assert np.allclose(vec1, vec2), "相同商品应产生相同向量"
    assert abs(np.linalg.norm(vec1) - 1.0) < 1e-6, "向量应归一化"
    print(f"  ✓ 商品向量维度={vec1.shape[0]}, L2范数={np.linalg.norm(vec1):.4f}")

    # ── Test 2：意图推断 ──
    print("\n[Test 2] 意图推断")
    p0 = MOTHER_BABY_PRODUCTS[0]
    intent0 = infer_intention(p0)
    assert len(intent0.intent_text) > 0, "意图文本不能为空"
    assert len(intent0.key_attribute) > 0, "关键属性不能为空"
    assert intent0.intent_vector.shape == (32,), "意图向量维度应为32"
    print(f"  ✓ 意图: {intent0.intent_text[:40]}...")
    print(f"  ✓ 关键属性: {intent0.key_attribute}")

    p1 = MOTHER_BABY_PRODUCTS[1]
    intent1 = infer_intention(p1, intent0)
    drift = intent1.drift_score(intent0)
    assert 0.0 <= drift <= 1.0, "漂移分数应在[0,1]范围内"
    print(f"  ✓ 意图漂移分数: {drift:.4f}")

    # ── Test 3：意图树构建 ──
    print("\n[Test 3] 会话意图轨迹构建")
    products_subset = MOTHER_BABY_PRODUCTS[:4]
    traj = build_session_intent_trajectory(
        session_id="test_session_001",
        user_id="test_user",
        products=products_subset,
    )
    assert len(traj.intentions) == len(products_subset), "意图数量应等于商品数量"
    assert all(i.step == s for s, i in enumerate(traj.intentions)), "步骤编号应正确"
    print(f"  ✓ 构建成功: {len(traj.products)} 个商品, {len(traj.intentions)} 个意图节点")

    # ── Test 4：Task4 探索评分 ──
    print("\n[Test 4] Task4 探索-利用评分")
    # 场景A：跨品类（应为探索型=3）
    cross_cat_products = [MOTHER_BABY_PRODUCTS[0], MOTHER_BABY_PRODUCTS[5]]  # Formula → Stroller
    traj_cross = build_session_intent_trajectory("s_cross", "u", cross_cat_products)
    score_cross = traj_cross.get_exploration_score()
    assert score_cross == 3, f"跨品类应为探索型(3)，实际={score_cross}"
    print(f"  ✓ 跨品类会话 → 探索评分={score_cross} (全局探索)")

    # 场景B：同品类配方奶粉（应为利用型=1或局部探索=2）
    same_cat_products = MOTHER_BABY_PRODUCTS[:3]  # 3个奶粉
    traj_same = build_session_intent_trajectory("s_same", "u", same_cat_products)
    score_same = traj_same.get_exploration_score()
    assert score_same in (1, 2), f"同品类应为利用型或局部探索，实际={score_same}"
    print(f"  ✓ 同品类会话 → 探索评分={score_same} (1=利用, 2=局部探索)")

    # ── Test 5：意图标签分类 ──
    print("\n[Test 5] 意图标签分类")
    label = classify_intent_label(traj, step=2)
    assert label in ("探索型", "比较型", "目标购买型", "复购型"), f"标签不合法: {label}"
    print(f"  ✓ 步骤2的意图标签: {label}")

    # ── Test 6：桑基图路径标注 ──
    print("\n[Test 6] 桑基图路径语义标注")
    all_trajs = simulate_pregnancy_to_postnatal_sessions(n_users=20, seed=0)
    assert len(all_trajs) > 0, "应生成至少一条轨迹"

    path_labels = label_sankey_paths(all_trajs, [])
    assert len(path_labels) > 0, "应能识别出至少一条路径"

    print(f"  ✓ 生成 {len(all_trajs)} 条轨迹，识别 {len(path_labels)} 条不同路径")
    for pl in path_labels[:3]:
        print(f"    {pl.from_page} → {pl.to_page}: "
              f"用户={pl.user_count}, 标签={pl.intent_label}, "
              f"漂移={pl.drift_score:.3f}, 关键属性={pl.key_attribute}")

    # ── Test 7：孕期→产后意图漂移场景 ──
    print("\n[Test 7] 孕期→产后意图漂移检测")
    # 构建孕期会话（婴儿车）和产后会话（奶粉+纸尿裤）
    pregnancy_products = MOTHER_BABY_PRODUCTS[4:7]   # Stroller
    postnatal_products = MOTHER_BABY_PRODUCTS[0:2] + MOTHER_BABY_PRODUCTS[7:9]  # Formula + Diaper

    traj_preg = build_session_intent_trajectory("s_preg", "u_preg", pregnancy_products)
    traj_post = build_session_intent_trajectory("s_post", "u_preg", postnatal_products)

    # 跨会话漂移：孕期最后意图 vs 产后首个意图
    if traj_preg.intentions and traj_post.intentions:
        cross_session_drift = traj_preg.intentions[-1].drift_score(traj_post.intentions[0])
        print(f"  ✓ 孕期→产后跨会话意图漂移: {cross_session_drift:.4f}")
        print(f"    孕期末意图: {traj_preg.intentions[-1].intent_text[:40]}...")
        print(f"    产后首意图: {traj_post.intentions[0].intent_text[:40]}...")
        # 跨品类漂移通常较高
        assert cross_session_drift > 0.1, "跨会话（不同需求阶段）漂移应可观测"
        print(f"  ✓ 漂移显著（>{0.1:.1f}），可识别营销时机窗口")

    print("\n" + "=" * 60)
    print("✅ 所有 7 个测试通过！")
    print("=" * 60)
    return True


# ─────────────────────────────────────────────
# 7. 主入口：完整演示
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # 运行测试
    run_tests()

    print("\n" + "=" * 60)
    print("🚀 SessionIntentBench 完整演示")
    print("=" * 60)

    # 生成模拟数据
    print("\n📦 生成孕期→产后用户会话数据...")
    trajectories = simulate_pregnancy_to_postnatal_sessions(n_users=100, seed=42)
    print(f"  共生成 {len(trajectories)} 条会话轨迹")

    # 桑基图路径标注
    print("\n🔍 桑基图路径语义标注...")
    path_labels = label_sankey_paths(trajectories, [])
    print(f"\n  发现 {len(path_labels)} 条不同品类转跳路径：")
    print(f"  {'路径':<35} {'用户数':>6} {'意图标签':<10} {'漂移':>6} {'关键属性'}")
    print("  " + "-" * 80)
    for pl in path_labels:
        print(f"  {pl.from_page[:16]:<16} → {pl.to_page[:16]:<16} "
              f"{pl.user_count:>6}  {pl.intent_label:<10} "
              f"{pl.drift_score:>6.3f}  {pl.key_attribute[:30]}")

    # 统计意图标签分布
    print("\n📊 意图标签分布统计：")
    label_dist: Dict[str, int] = {}
    total_users = sum(pl.user_count for pl in path_labels)
    for pl in path_labels:
        label_dist[pl.intent_label] = label_dist.get(pl.intent_label, 0) + pl.user_count
    for label, count in sorted(label_dist.items(), key=lambda x: -x[1]):
        pct = count / total_users * 100 if total_users > 0 else 0
        print(f"  {label:<12}: {count:>5} 次转跳 ({pct:.1f}%)")

    print("\n💡 业务洞察：")
    for pl in path_labels[:3]:
        if pl.intent_label == "探索型":
            print(f"  🔍 {pl.from_page}→{pl.to_page}: 探索型用户需要内容引导（推荐博客/评测）")
        elif pl.intent_label == "比较型":
            print(f"  ⚖️  {pl.from_page}→{pl.to_page}: 比较型用户需要对比表（突出差异化优势）")
        elif pl.intent_label == "目标购买型":
            print(f"  🎯 {pl.from_page}→{pl.to_page}: 目标购买型用户减少摩擦（直接展示加购按钮）")
        elif pl.intent_label == "复购型":
            print(f"  🔄 {pl.from_page}→{pl.to_page}: 复购型用户推送订阅优惠（提升LTV）")
```

---

## ④ 技能关联

| 关系 | 技能 | 理由 |
|------|------|------|
| 前置 | [Skill-Customer-Journey-Prototype](../06-增长模型/Skill-Customer-Journey-Prototype.md) | 旅程原型识别是意图漂移检测的基础；需先理解用户旅程框架，才能定义"漂移"的参考基准 |
| 前置 | [Skill-User-Lifecycle-STAN](../06-增长模型/Skill-User-Lifecycle-STAN.md) | 生命周期阶段（孕期/新生儿期/成长期）与意图漂移高度相关；生命周期分类可作为意图漂移的先验条件 |
| 组合 | [Skill-TRACE-Clickstream-Embedding](./Skill-TRACE-Clickstream-Embedding.md) | Clickstream 嵌入+意图标签=完整session理解；TRACE 负责建模"用户去哪了"，SessionIntentBench 负责解释"用户为何这样走" |
| 组合 | [Skill-Trajectory-Pattern-Mining](./Skill-Trajectory-Pattern-Mining.md) | 轨迹模式挖掘提供路径骨架，意图语义提供路径叙事；两者结合=桑基图的完整故事 |
| 延伸 | [Skill-NonItem-Page-Path-Modeling](./Skill-NonItem-Page-Path-Modeling.md) | 非商品页（首页/分类页/博客页）的路径建模与意图漂移互为补充，覆盖完整的用户决策过程 |

---

- **前置技能**：[[Skill-TRACE-Clickstream-Embedding]] | [[Skill-Trajectory-Pattern-Mining]]
- **延伸技能**：[[Skill-PersonaBot-RAG-Profiling]]
- **可组合技能**：[[Skill-RFM-User-Segmentation]] | [[Skill-Cohort-Retention-Analysis]]

## ⑤ 商业价值评估

| 维度 | 评分 | 依据 |
|------|------|------|
| ROI 预估 | ⭐⭐⭐☆☆ | 意图标注为桑基图增加叙事价值，可驱动页面优化和内容策略；直接货币化路径长（需结合 A/B 实验验证优化效果）；预计间接提升关键路径转化 **10-15%** |
| 实施难度 | ⭐⭐⭐☆☆ | 规则+embedding方案成本极低可快速验证；若使用 LLM 意图推断（GPT-4o-mini）成本约 $0.001/session；主要挑战在于标注质量和意图类别定义的业务对齐 |
| 优先级 | ⭐⭐☆☆☆ | P2级别：锦上添花型需求；母婴业务会话数据完整（有商品 title/price/category）可直接套用；适合在桑基图基础分析完成后，作为语义增强层叠加 |
| 数据门槛 | ⭐⭐☆☆☆ | 仅需商品级别会话序列（title + category + price），无需用户个人信息；Amazon M2 公开数据可直接测试，迁移到自有数据改动极小 |
| 可复用性 | ⭐⭐⭐⭐☆ | 意图分类体系（探索/比较/目标购买/复购）适用于所有垂类电商；意图树构建流程对任何有 session 数据的场景均可复用 |

**推荐使用场景优先级：**
1. 🥇 **桑基图语义标注**：为现有流量路径分析叠加意图标签，零额外数据需求，直接提升洞察深度
2. 🥈 **孕期→产后意图漂移监测**：在关键生命周期转折点识别营销时机，精准推送产后必备品
3. 🥉 **推荐策略切换依据**：Task4 探索评分作为"何时利用/何时探索"的决策信号，指导推荐系统策略切换
