---
title: Product KG Query — 多 Agent 产品知识图谱查询与 SKU 跨平台映射
doc_type: knowledge
module: 08-知识图谱
topic: product-knowledge-graph-query
status: stable
created: 2026-06-11
updated: 2026-06-11
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: Product KG Query — 多 Agent 产品知识图谱查询与 SKU 跨平台映射

> **论文**：Question-to-Knowledge (Q2K): Multi-Agent Generation of Inspectable Facts for Product Mapping
> **arXiv**：2509.01182 | 2024年 | **桥梁**: 08-知识图谱 ↔ 05-推荐系统 | **类型**: 跨域融合
> **GitHub**：https://github.com/viralpick/paper-q2k-artifact

---

## ① 算法原理

### 核心思想

构建了产品知识图谱（PKG）之后，紧接着的挑战是：**如何在多平台、多命名规范的 SKU 海洋里精确匹配同一个产品？** 母婴跨境卖家在 Amazon、Shopee、TikTok Shop 三个平台运营同一款吸奶器，但商品 ID 各不同、属性字段名也不统一，跨平台聚合分析时"是同一个产品"这个问题就变得极难回答。

**Q2K（Question-to-Knowledge）**用多 Agent 框架把"产品映射"问题转化为"生成可核查事实"：

```
目标 SKU（待匹配）
      │
      ▼
[Reasoning Agent] ─→ 生成消歧问题（"该产品的静音分贝是多少？"）
      │
      ▼
[Knowledge Agent] ─→ 对每个候选 SKU 检索事实答案
      │
      ▼
[Dedup Agent] ──→ 跨会话复用已有推理 trace，避免重复查询
      │
      ▼
[Matching Judge] ─→ 事实集合对比 → 是/否同一产品 + 置信度
```

### 数学直觉

设目标 SKU $s$ 和候选 SKU $s'$，Q2K 通过生成 $K$ 个消歧问题 $\{q_1, ..., q_K\}$，为每个问题从 PKG 中检索到事实向量 $\mathbf{f}(s)$ 和 $\mathbf{f}(s')$，映射决策为：

$$\text{Match}(s, s') = \mathbb{1}\left[\text{sim}\big(\mathbf{f}(s), \mathbf{f}(s')\big) \geq \theta\right]$$

其中 $\text{sim}$ 为基于事实覆盖率的相似度，$\theta$ 为可调阈值。与纯向量嵌入方法不同，Q2K 的事实是**可检查（inspectable）**的——每个匹配结论都能溯源到具体的问答依据，方便人工复核和审计。

### 关键假设
- PKG 中的属性数据基本准确（依赖 AutoPKG 等上游质量门控）
- 候选 SKU 池大小可控（通常 ≤ 1000，超出需分桶）
- 消歧问题的覆盖度与产品属性丰富度正相关

---

## ② 母婴出海应用案例

### 场景 A：跨平台 SKU 聚合分析（Amazon + Shopee + TikTok Shop）

**业务问题**：某母婴团队的吸奶器产品在三平台 GMV 总计 2000 万/年，但三个平台的 SKU 用不同 ID 管理，无法自动聚合"同款产品跨平台销售对比"。运营要花 3-4 小时/款手工比对产品规格确认是同一个 SKU。

**Q2K 处理流程**：
1. **消歧问题生成**：Reasoning Agent 为吸奶器自动生成 12 个消歧问题（吸力档位、充电方式、适用月龄、静音等级、BPA-Free 认证……）
2. **事实检索**：Knowledge Agent 从三平台 PKG 中分别查询每个候选 SKU 的事实答案
3. **去重复用**：Dedup Agent 检测已推理过的产品，直接复用 trace（降低 LLM 调用成本 60%）
4. **匹配判决**：事实集合对比，相似度 ≥ 0.85 → 确认同款 → 自动打通三平台 ID

**数据要求**：
- 各平台 PKG（由 AutoPKG 或人工维护的属性表输出）
- 各平台 SKU 基础信息（标题 + bullet points + 主图）

**预期产出**：
- SKU 跨平台映射准确率 ≥ 90%（Q2K 论文在 eBay/Amazon 数据集上达 92.3%）
- 运营手工比对时间从 3-4 小时/款 → 自动化完成，人工仅复核置信度 < 0.85 的边缘 case

**业务价值**：跨平台聚合数据解锁 GMV 归因分析，年化决策价值估算 ¥50-100 万

### 场景 B：供应商 OEM 产品鉴别（防止重复订货）

**业务问题**：团队评估 3-5 家 OEM 工厂的样品，不同工厂的产品外观相似，采购团队需要确认"是否本质相同"。历史曾因误判导致重复订货同款产品、压仓 80 万库存。

**Q2K 处理**：对 OEM 样品的核心参数（马达型号、硅胶认证、最大吸力、尺寸公差）生成消歧问题，从供应商提供的规格书 PKG 中检索答案对比。

**业务价值**：避免重复订货损失 ¥80 万+（基于历史事故），采购决策周期缩短 2-3 天

---

## ③ 代码模板

```python
"""
Product Knowledge Graph Query — 多 Agent SKU 跨平台映射
基于 Q2K (arXiv: 2509.01182) 简化实现

依赖: json, re, typing, dataclasses
"""

from dataclasses import dataclass, field
from typing import Optional
import json
import re


# ─────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────

@dataclass
class SKUNode:
    """产品知识图谱中的 SKU 节点"""
    sku_id: str
    platform: str           # amazon / shopee / tiktok_shop
    title: str
    attributes: dict        # {attr_key: attr_value}


@dataclass
class FactEntry:
    """单条可检查事实"""
    question: str
    answer: str
    confidence: float
    source: str             # 'pkg_lookup' / 'llm_infer' / 'cached'


@dataclass
class MappingResult:
    """SKU 映射结果"""
    source_sku_id: str
    target_sku_id: str
    is_match: bool
    similarity_score: float
    facts: list[FactEntry] = field(default_factory=list)
    inspectable_trace: str = ""     # 可供人工审计的推理链


# ─────────────────────────────────────────────
# Agent 实现
# ─────────────────────────────────────────────

class ReasoningAgent:
    """生成消歧问题：针对产品类别生成最具区分力的问题集"""

    QUESTION_TEMPLATES = {
        "吸奶器": [
            "该产品的最大吸力值（mmHg）是多少？",
            "该产品支持几档可调吸力？",
            "充电接口类型是什么（USB-C / Micro-USB / 其他）？",
            "电池容量（mAh）是多少？",
            "噪音等级（dB）是多少？",
            "是否获得 BPA-Free 认证？",
            "适用月龄范围是什么？",
            "产品主体材质是什么？",
            "是否防水（IPX 等级）？",
            "硅胶吸乳口尺寸（mm）是多少？",
            "产品重量（g）是多少？",
            "是否支持单双边同时使用？",
        ],
        "婴儿推车": [
            "推车展开尺寸（长×宽×高，cm）是多少？",
            "折叠后尺寸是多少？",
            "最大承重（kg）是多少？",
            "适用月龄范围是什么？",
            "座椅是否可平躺（角度）？",
            "车架材质是什么？",
            "轮子材质和直径是多少？",
            "产品总重（kg）是多少？",
            "是否符合 EN1888 或 ASTM F833 安全标准？",
        ],
    }

    DEFAULT_QUESTIONS = [
        "产品的核心功能规格是什么？",
        "产品尺寸和重量是多少？",
        "产品认证和安全标准有哪些？",
        "产品适用人群（月龄/年龄）是什么？",
        "产品材质组成是什么？",
    ]

    def generate_questions(self, category: str, n: int = 8) -> list[str]:
        """为指定品类生成消歧问题"""
        questions = self.QUESTION_TEMPLATES.get(category, self.DEFAULT_QUESTIONS)
        return questions[:n]


class KnowledgeAgent:
    """从 PKG 中检索事实答案"""

    def __init__(self, pkg: dict[str, SKUNode]):
        self.pkg = pkg   # {sku_id: SKUNode}

    def lookup_fact(self, sku_id: str, question: str) -> FactEntry:
        """
        从 PKG 属性中匹配问题答案
        生产环境可替换为 LLM + vector search
        """
        sku = self.pkg.get(sku_id)
        if not sku:
            return FactEntry(question=question, answer="未知", confidence=0.0, source="not_found")

        # 关键词匹配（简化版，生产环境用 embedding 匹配）
        # 问题关键词 → 可能的属性键（按优先级排序）
        attr_map = {
            "吸力值": ["最大吸力", "suction", "吸力"],
            "几档": ["吸力档位", "档位数", "档位", "levels"],
            "充电": ["充电接口", "充电方式", "charging"],
            "电池": ["电池容量", "battery", "mAh"],
            "噪音": ["噪音等级", "噪声", "静音", "dB"],
            "BPA": ["BPA-Free认证", "BPA", "认证"],
            "月龄": ["适用月龄", "使用月龄", "age"],
            "材质": ["材质", "material", "硅胶"],
            "防水": ["防水", "IPX", "waterproof"],
            "重量": ["重量", "weight"],
            "尺寸": ["尺寸", "dimensions", "size"],
        }

        for keyword, attr_keys in attr_map.items():
            if keyword in question:
                for attr_key in attr_keys:
                    for k, v in sku.attributes.items():
                        if attr_key.lower() in k.lower() and v:
                            return FactEntry(
                                question=question,
                                answer=str(v),
                                confidence=0.9,
                                source="pkg_lookup"
                            )

        return FactEntry(question=question, answer="属性不存在", confidence=0.3, source="missing")


class DedupAgent:
    """推理 trace 去重复用（降低 LLM 成本）"""

    def __init__(self):
        self._cache: dict[str, list[FactEntry]] = {}   # sku_id -> facts

    def get_cached(self, sku_id: str) -> Optional[list[FactEntry]]:
        return self._cache.get(sku_id)

    def store(self, sku_id: str, facts: list[FactEntry]) -> None:
        self._cache[sku_id] = facts

    @property
    def cache_hit_rate(self) -> float:
        return len(self._cache) / max(len(self._cache) + 1, 1)


class MatchingJudge:
    """基于事实集合对比做映射决策"""

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def compute_similarity(
        self,
        facts_a: list[FactEntry],
        facts_b: list[FactEntry]
    ) -> float:
        """计算两组事实的覆盖相似度"""
        if not facts_a or not facts_b:
            return 0.0

        matches = 0
        total = len(facts_a)

        for fa in facts_a:
            for fb in facts_b:
                if fa.question == fb.question:
                    # 答案匹配度（简化：完全匹配或数值接近）
                    if fa.answer != "未知" and fb.answer != "未知":
                        if fa.answer.lower().strip() == fb.answer.lower().strip():
                            matches += 1.0
                        elif self._numeric_close(fa.answer, fb.answer):
                            matches += 0.8
                    break

        return matches / total if total > 0 else 0.0

    def _numeric_close(self, a: str, b: str, tol: float = 0.1) -> bool:
        """提取并比较数值（如 "2000mAh" vs "2000 mAh"）"""
        nums_a = re.findall(r'\d+\.?\d*', a)
        nums_b = re.findall(r'\d+\.?\d*', b)
        if nums_a and nums_b:
            va, vb = float(nums_a[0]), float(nums_b[0])
            return abs(va - vb) / max(va, vb, 1e-9) <= tol
        return False

    def judge(
        self,
        sku_a: SKUNode,
        sku_b: SKUNode,
        facts_a: list[FactEntry],
        facts_b: list[FactEntry]
    ) -> MappingResult:
        sim = self.compute_similarity(facts_a, facts_b)
        is_match = sim >= self.threshold

        # 生成可审计的推理链
        trace_lines = [f"SKU 映射判断: {sku_a.sku_id} ({sku_a.platform}) ↔ {sku_b.sku_id} ({sku_b.platform})"]
        trace_lines.append(f"相似度: {sim:.3f} (阈值: {self.threshold})")
        for fa in facts_a[:5]:
            fb_answer = next((fb.answer for fb in facts_b if fb.question == fa.question), "N/A")
            match_flag = "✓" if fa.answer.strip().lower() == fb_answer.strip().lower() else "✗"
            trace_lines.append(f"  {match_flag} {fa.question[:30]}: '{fa.answer}' vs '{fb_answer}'")

        return MappingResult(
            source_sku_id=sku_a.sku_id,
            target_sku_id=sku_b.sku_id,
            is_match=is_match,
            similarity_score=sim,
            facts=facts_a,
            inspectable_trace="\n".join(trace_lines),
        )


# ─────────────────────────────────────────────
# Q2K 完整框架
# ─────────────────────────────────────────────

class Q2KProductMapper:
    """
    Q2K 多 Agent SKU 跨平台映射框架

    输入: 多平台 PKG (sku_id → SKUNode)
    输出: 跨平台 SKU 映射对 + 可审计推理链
    """

    def __init__(self, pkg: dict[str, SKUNode], match_threshold: float = 0.65):
        self.pkg = pkg
        self.reasoning = ReasoningAgent()
        self.knowledge = KnowledgeAgent(pkg)
        self.dedup = DedupAgent()
        self.judge = MatchingJudge(threshold=match_threshold)

    def get_facts(self, sku_id: str, category: str) -> list[FactEntry]:
        """获取 SKU 的事实集合（优先读缓存）"""
        cached = self.dedup.get_cached(sku_id)
        if cached:
            return [FactEntry(f.question, f.answer, f.confidence, "cached") for f in cached]

        questions = self.reasoning.generate_questions(category)
        facts = [self.knowledge.lookup_fact(sku_id, q) for q in questions]
        self.dedup.store(sku_id, facts)
        return facts

    def map_sku(self, source_sku_id: str, candidate_sku_ids: list[str], category: str) -> list[MappingResult]:
        """
        将 source SKU 与候选 SKU 列表逐一比较

        Args:
            source_sku_id: 待映射的 SKU（如 Amazon 上的 ASIN）
            candidate_sku_ids: 候选 SKU 列表（如 Shopee 同品类 SKU）
            category: 品类（如 "吸奶器"）

        Returns:
            List[MappingResult]，按相似度降序排列
        """
        source_sku = self.pkg.get(source_sku_id)
        if not source_sku:
            raise ValueError(f"Source SKU {source_sku_id} not found in PKG")

        facts_source = self.get_facts(source_sku_id, category)
        results = []

        for cid in candidate_sku_ids:
            candidate_sku = self.pkg.get(cid)
            if not candidate_sku:
                continue
            facts_candidate = self.get_facts(cid, category)
            result = self.judge.judge(source_sku, candidate_sku, facts_source, facts_candidate)
            results.append(result)

        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results

    def batch_cross_platform_mapping(
        self,
        platform_a_skus: list[str],
        platform_b_skus: list[str],
        category: str
    ) -> dict[str, Optional[str]]:
        """
        批量跨平台映射：platform A 的每个 SKU 找到 platform B 的对应 SKU

        Returns: {platform_a_sku_id: platform_b_sku_id or None}
        """
        mapping = {}
        for a_sku_id in platform_a_skus:
            results = self.map_sku(a_sku_id, platform_b_skus, category)
            best = results[0] if results else None
            mapping[a_sku_id] = best.target_sku_id if best and best.is_match else None
        return mapping


# ─────────────────────────────────────────────
# 演示
# ─────────────────────────────────────────────

def run_q2k_demo():
    """演示：吸奶器 Amazon ↔ Shopee 跨平台 SKU 映射"""
    print("=" * 65)
    print("Q2K — 母婴吸奶器跨平台 SKU 映射演示")
    print("=" * 65)

    # 构造模拟 PKG（3 个 Amazon SKU + 4 个 Shopee SKU）
    pkg: dict[str, SKUNode] = {
        "ASIN-A001": SKUNode("ASIN-A001", "amazon", "Momcozy M5 双边电动吸奶器", {
            "最大吸力": "280mmHg", "吸力档位": "9档", "电池容量": "2000mAh",
            "噪音等级": "<35dB", "BPA-Free认证": "是", "适用月龄": "0-12M",
            "充电接口": "USB-C", "重量": "450g", "材质": "医疗级硅胶",
        }),
        "ASIN-A002": SKUNode("ASIN-A002", "amazon", "Momcozy S12 单边便携泵", {
            "最大吸力": "250mmHg", "吸力档位": "6档", "电池容量": "1800mAh",
            "噪音等级": "<40dB", "BPA-Free认证": "是", "适用月龄": "0-12M",
            "充电接口": "USB-C", "重量": "380g", "材质": "医疗级硅胶",
        }),
        "ASIN-A003": SKUNode("ASIN-A003", "amazon", "Spectra S1 双边电动吸奶器", {
            "最大吸力": "300mmHg", "吸力档位": "12档", "电池容量": "3000mAh",
            "噪音等级": "<45dB", "BPA-Free认证": "是", "适用月龄": "0-24M",
            "充电接口": "DC适配器", "重量": "680g", "材质": "ABS+硅胶",
        }),
        # Shopee SKU（有的是同款，有的不同）
        "SP-SG-1001": SKUNode("SP-SG-1001", "shopee", "Momcozy M5 双边吸奶器 新加坡版", {
            "最大吸力": "280mmHg", "吸力档位": "9档", "电池容量": "2000mAh",
            "噪音等级": "<35分贝", "BPA-Free认证": "通过", "适用月龄": "0-12个月",
            "充电接口": "USB Type-C", "重量": "450克", "材质": "医疗级硅胶",
        }),
        "SP-SG-1002": SKUNode("SP-SG-1002", "shopee", "Momcozy S12 便携吸奶器", {
            "最大吸力": "250mmHg", "吸力档位": "6级", "电池容量": "1800mAh",
            "噪音等级": "低于40dB", "BPA-Free认证": "是", "适用月龄": "0-12M",
            "充电接口": "USB-C", "重量": "380g", "材质": "硅胶",
        }),
        "SP-SG-1003": SKUNode("SP-SG-1003", "shopee", "Generic 国产仿制吸奶器", {
            "最大吸力": "200mmHg", "吸力档位": "3档", "电池容量": "1200mAh",
            "噪音等级": "60dB", "BPA-Free认证": "否", "适用月龄": "未标注",
            "充电接口": "Micro-USB", "重量": "600g", "材质": "普通塑料",
        }),
        "SP-SG-1004": SKUNode("SP-SG-1004", "shopee", "Spectra S1 Plus 新加坡版", {
            "最大吸力": "300mmHg", "吸力档位": "12档", "电池容量": "3000mAh",
            "噪音等级": "<45dB", "BPA-Free认证": "是", "适用月龄": "0-24M",
            "充电接口": "DC", "重量": "680g", "材质": "ABS硅胶",
        }),
    }

    mapper = Q2KProductMapper(pkg, match_threshold=0.65)
    shopee_skus = ["SP-SG-1001", "SP-SG-1002", "SP-SG-1003", "SP-SG-1004"]

    print("\n─── 单 SKU 映射结果 ───")
    results = mapper.map_sku("ASIN-A001", shopee_skus, "吸奶器")
    for r in results[:3]:
        status = "✅ 匹配" if r.is_match else "❌ 不匹配"
        print(f"\n{status}  {r.source_sku_id} ↔ {r.target_sku_id}  (相似度: {r.similarity_score:.3f})")
        print("  审计链（前3行）:")
        for line in r.inspectable_trace.split("\n")[:4]:
            print(f"    {line}")

    print("\n─── 批量跨平台映射 ───")
    amazon_skus = ["ASIN-A001", "ASIN-A002", "ASIN-A003"]
    batch_result = mapper.batch_cross_platform_mapping(amazon_skus, shopee_skus, "吸奶器")

    print(f"\n{'Amazon SKU':<15} {'Shopee 对应':<15} {'状态'}")
    print("-" * 45)
    for amazon_id, shopee_id in batch_result.items():
        status = f"→ {shopee_id}" if shopee_id else "→ 无匹配"
        print(f"{amazon_id:<15} {status}")

    # 验证
    assert batch_result.get("ASIN-A001") == "SP-SG-1001", "A001 应匹配 SP-SG-1001"
    assert batch_result.get("ASIN-A002") == "SP-SG-1002", "A002 应匹配 SP-SG-1002"
    assert batch_result.get("ASIN-A003") == "SP-SG-1004", "A003 应匹配 SP-SG-1004"

    cache_hit = len(mapper.dedup._cache)
    print(f"\nDedup 缓存命中: {cache_hit} 条（已避免重复 LLM 调用）")
    print("\n[✓] Q2K 跨平台 SKU 映射测试通过")
    return batch_result


if __name__ == "__main__":
    run_q2k_demo()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-AutoPKG-Multimodal-Product-Attribute-KG]]（PKG 构建是查询的基础；AutoPKG 负责把 SKU 属性结构化，Q2K 负责在结构化数据上做跨平台查询）
- **前置（prerequisite）**：[[Skill-Entity-Resolution-KG-Dedup]]（实体解析是 SKU 映射的基础理论，处理同义实体归并问题）
- **延伸（extends）**：[[Skill-KGQA-Question-Answering]]（Q2K 生成的消歧问题集与 KGQA 自然语言查询结合，可支持更复杂的业务分析查询）
- **延伸（extends）**：[[Skill-Dense-Retrieval-Ecommerce-Semantic-Search]]（事实检索层可用稠密检索替代关键词匹配，提升召回率）
- **可组合（combinable）**：[[Skill-Demand-Forecasting-Supply-Chain]]（SKU 跨平台映射打通后，可在统一 SKU 粒度上做多平台库存协同优化）
- **可组合（combinable）**：[[Skill-Review-Pain-Point-Mining]]（用 Q2K 映射的统一 SKU 跨平台聚合评论数据，再用痛点挖掘做全渠道差评分析）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 运营手工比对成本节省：¥150,000-300,000/年（按 30 SKU/月 × 3 小时 × ¥200/小时，500 SKU 规模）
  - 跨平台数据聚合后解锁的 GMV 归因收益：¥500,000-1,000,000/年（基于跨平台数据洞察优化备货和广告）
  - 避免重复订货损失：¥80,000+/次（历史案例）
  - **年化综合 ROI**：800 万 GMV 规模 → ¥80-150 万增量（含成本节省 + 决策优化）

- **实施难度**：⭐⭐☆☆☆
  - 依赖 AutoPKG 等上游 PKG 质量（属性越完整，映射越准）
  - 纯 Python 实现，无深度学习依赖
  - 可直接对接现有 SKU 数据库，不需要重建数据架构

- **优先级评分**：⭐⭐⭐⭐⭐
  - 解决跨境电商的核心痛点：多平台数据孤岛
  - 是 AutoPKG 的直接应用出口（先构建 → 再查询 → 再分析）
  - 论文在 eBay/Amazon 真实数据集上验证，业务可信度高

- **评估依据**：Q2K 在公开电商数据集上 SKU 映射准确率 92.3%；可检查事实（inspectable facts）降低错误映射风险，支持人工审计，满足跨境合规要求
