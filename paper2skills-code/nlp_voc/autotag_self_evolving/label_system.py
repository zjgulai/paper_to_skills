"""标签体系管理模块

管理 L1-L4 层级标签的树状结构，支持 CRUD、层级查询、路径追踪。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ── TELEClass 辅助函数 ─────────────────────────────────────────

_STOPWORDS = {
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "那", "啊", "哦", "呢", "吧", "吗", "么", "之", "与", "及", "等",
    "从", "向", "往", "于", "而", "但", "因为", "所以", "如果", "虽然", "或者", "还是",
    "这个", "那个", "什么", "怎么", "为什么", "如何", "可以", "可能", "应该", "觉得",
    "感觉", "认为", "好像", "应该", "需要", "想要", "希望", "已经", "正在", "曾经",
    "时候", "时间", "地方", "东西", "事情", "问题", "情况", "原因", "结果", "方式",
    "非常", "特别", "比较", "相当", "真的", "确实", "简直", "实在", "大概", "大约",
    "一直", "总是", "经常", "常常", "有时", "偶尔", "很少", "几乎", "完全", "全部",
    "所有", "一切", "大家", "别人", "有的", "一些", "一下", "一点", "一次", "一天",
    "一次", "第一", "最后", "中间", "之前", "之后", "以前", "以后", "现在", "当时",
    "孩子", "宝宝", "小孩", "儿子", "女儿", "妈妈", "爸爸", "家里", "家里", "家里",
    "就是", "还是", "不过", "然后", "接着", "后来", "同时", "另外", "此外", "而且",
    "但是", "然而", "尽管", "即使", "无论", "不管", "只要", "只有", "除非", "除了",
    "为了", "关于", "根据", "按照", "通过", "经过", "随着", "除了", "除去", "作为",
    "product", "item", "buy", "purchase", "order", "ship", "shipping", "receive",
    "got", "get", "use", "using", "used", "would", "could", "should", "will", "can",
    "one", "two", "first", "last", "also", "really", "very", "quite", "pretty",
    "much", "many", "more", "most", "some", "any", "no", "not", "good", "bad",
    "nice", "great", "well", "like", "love", "hate", "dislike", "recommend",
    "satisfied", "dissatisfied", "happy", "unhappy", "pleased", "disappointed",
    "again", "back", "return", "exchange", "refund", "money", "price", "cheap",
    "expensive", "worth", "value", "quality", "quality", "cheap", "fast", "slow",
}


def _extract_candidates(text: str) -> list[str]:
    """从文本中提取候选关键词：中文 2-4 字词 + 英文单词。"""
    candidates = []

    # 中文 2-4 字词（连续中文字符）
    for length in range(2, 5):
        for i in range(len(text) - length + 1):
            substr = text[i:i + length]
            if all("\u4e00" <= c <= "\u9fff" for c in substr):
                candidates.append(substr)

    # 英文单词（2+ 字符）
    for m in re.finditer(r"[a-z]{2,}", text):
        candidates.append(m.group())

    return candidates


@dataclass
class LabelNode:
    """单个标签节点"""

    id: str                          # 唯一标识，如 "L1-001", "L4-023"
    name: str                        # 标签名称，如 "纸尿裤", "夜间侧漏"
    level: int                       # 层级 1-4
    parent_id: Optional[str] = None  # 父标签ID
    description: str = ""            # 标签描述
    keywords: list[str] = field(default_factory=list)  # 触发关键词
    status: str = "active"           # active / dormant / candidate
    created_at: str = ""             # 创建时间
    hit_count: int = 0               # 命中次数（用于淘汰判断）
    last_hit_at: str = ""            # 最后一次命中时间

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "level": self.level,
            "parent_id": self.parent_id,
            "description": self.description,
            "keywords": self.keywords,
            "status": self.status,
            "created_at": self.created_at,
            "hit_count": self.hit_count,
            "last_hit_at": self.last_hit_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> LabelNode:
        return cls(
            id=d["id"],
            name=d["name"],
            level=d["level"],
            parent_id=d.get("parent_id"),
            description=d.get("description", ""),
            keywords=d.get("keywords", []),
            status=d.get("status", "active"),
            created_at=d.get("created_at", ""),
            hit_count=d.get("hit_count", 0),
            last_hit_at=d.get("last_hit_at", ""),
        )


class LabelSystem:
    """标签体系管理器

    管理 L1-L4 四级标签的树状结构。

    Attributes:
        nodes: 所有标签节点的字典 {node_id: LabelNode}
        root_ids: L1 标签ID列表
    """

    def __init__(self) -> None:
        self.nodes: dict[str, LabelNode] = {}
        self._children_cache: dict[str, list[str]] = {}  # parent_id -> [child_ids]

    # ── 增删改查 ───────────────────────────────────────────────

    def add(self, node: LabelNode) -> None:
        """添加标签节点"""
        if node.id in self.nodes:
            raise ValueError(f"标签ID已存在: {node.id}")
        if node.level < 1 or node.level > 4:
            raise ValueError(f"层级必须在1-4之间: {node.level}")
        if node.level > 1 and not node.parent_id:
            raise ValueError(f"L{node.level} 标签必须指定 parent_id")
        if node.parent_id and node.parent_id not in self.nodes:
            raise ValueError(f"父标签不存在: {node.parent_id}")
        if node.parent_id:
            parent = self.nodes[node.parent_id]
            if parent.level != node.level - 1:
                raise ValueError(
                    f"层级不连续: 父标签是L{parent.level}, 子标签是L{node.level}"
                )

        self.nodes[node.id] = node
        self._invalidate_cache()

    def remove(self, node_id: str) -> None:
        """删除标签（级联删除子标签）"""
        if node_id not in self.nodes:
            raise ValueError(f"标签不存在: {node_id}")

        # 递归删除子标签
        children = self.get_children(node_id)
        for child in children:
            self.remove(child.id)

        del self.nodes[node_id]
        self._invalidate_cache()

    def get(self, node_id: str) -> LabelNode:
        """获取标签节点"""
        if node_id not in self.nodes:
            raise ValueError(f"标签不存在: {node_id}")
        return self.nodes[node_id]

    def get_children(self, parent_id: str) -> list[LabelNode]:
        """获取直接子标签"""
        if parent_id not in self._children_cache:
            self._children_cache[parent_id] = [
                nid for nid, n in self.nodes.items() if n.parent_id == parent_id
            ]
        return [self.nodes[nid] for nid in self._children_cache[parent_id]]

    def get_path(self, node_id: str) -> list[LabelNode]:
        """获取从根到该节点的完整路径"""
        path = []
        current_id: Optional[str] = node_id
        while current_id:
            node = self.nodes[current_id]
            path.append(node)
            current_id = node.parent_id
        return list(reversed(path))

    def get_path_names(self, node_id: str) -> dict[str, str]:
        """获取路径上的标签名称"""
        path = self.get_path(node_id)
        result: dict[str, str] = {}
        for node in path:
            result[f"l{node.level}"] = node.name
        return result

    def get_all_active(self) -> list[LabelNode]:
        """获取所有活跃标签"""
        return [n for n in self.nodes.values() if n.status == "active"]

    def get_leaves(self) -> list[LabelNode]:
        """获取所有叶子节点（无子标签的节点）"""
        has_children = {n.parent_id for n in self.nodes.values() if n.parent_id}
        return [n for nid, n in self.nodes.items() if nid not in has_children]

    # ── 搜索匹配 ───────────────────────────────────────────────

    def match_by_keyword(self, text: str) -> list[tuple[LabelNode, float]]:
        """根据关键词匹配标签，返回 (节点, 匹配分数) 列表"""
        text_lower = text.lower()
        results: list[tuple[LabelNode, float]] = []

        for node in self.nodes.values():
            if node.status != "active":
                continue
            score = 0.0
            for kw in node.keywords:
                if kw.lower() in text_lower:
                    score += len(kw) / len(text_lower)  # 简单长度加权
            if score > 0:
                results.append((node, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def find_by_name(self, name: str) -> Optional[LabelNode]:
        """根据名称查找标签"""
        for node in self.nodes.values():
            if node.name == name:
                return node
        return None

    # ── 序列化 ───────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LabelSystem:
        system = cls()
        for nid, node_dict in data.get("nodes", {}).items():
            system.nodes[nid] = LabelNode.from_dict(node_dict)
        return system

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, path: str) -> LabelSystem:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    # ── TELEClass 最小监督引导 ───────────────────────────────────

    def teleclass_bootstrap(
        self,
        categories: dict[str, list[str]],
        corpus: list[str],
        expander: Callable[[str, list[str]], list[str]] | None = None,
        max_l2_per_l1: int = 5,
        min_df: int = 3,
        window_size: int = 10,
    ) -> dict[str, list[str]]:
        """基于 TELEClass 思想，从种子词 + 无标注语料自动构建 L1-L2 标签体系。

        Args:
            categories: {L1 类别名: [种子关键词]}，如 {"纸尿裤": ["漏尿", "红屁股"]}
            corpus: 无标注文本语料列表（每条为一个文档/评论）
            expander: 可选的外部关键词扩展函数 (category_name, seeds) -> expanded_keywords
            max_l2_per_l1: 每个 L1 下最多发现的 L2 子类别数
            min_df: 候选关键词最小文档频率（过滤极低频噪声）
            window_size: 共现窗口大小（字符数），种子词附近多大范围统计共现词

        Returns:
            构建日志：{L1 名: [发现的 L2 名列表]}
        """
        if not corpus:
            raise ValueError("corpus 不能为空")

        built_log: dict[str, list[str]] = {}
        doc_lower = [d.lower() for d in corpus]

        for idx, (l1_name, seeds) in enumerate(categories.items(), start=1):
            # 1. 可选：LLM 扩展种子词
            keywords = list(set(seeds))
            if expander:
                try:
                    extended = expander(l1_name, seeds)
                    keywords = list(set(keywords + extended))
                except Exception:
                    pass  # 扩展失败时回退到原始种子词

            # 2. 在语料中定位种子词出现的文档和位置
            seed_positions = []  # [(doc_idx, pos)]
            for di, doc in enumerate(doc_lower):
                for seed in keywords:
                    pos = doc.find(seed.lower())
                    if pos != -1:
                        seed_positions.append((di, pos, seed))

            if len(seed_positions) < min_df:
                # 种子词出现太少，跳过该类别
                continue

            # 3. 共现窗口内提取候选词（简单分词：2-4 字中文词 + 英文单词）
            cooccur_counter: Counter = Counter()
            for di, pos, seed in seed_positions:
                doc = doc_lower[di]
                start = max(0, pos - window_size)
                end = min(len(doc), pos + len(seed) + window_size)
                window = doc[start:end]

                # 提取候选词：中文 2-4 字词 + 英文单词
                candidates = _extract_candidates(window)
                for cand in candidates:
                    if cand != seed.lower() and len(cand) >= 2:
                        cooccur_counter[cand] += 1

            # 4. 过滤低频候选词，取 top-k 作为 L2 类别
            filtered = {
                w: c for w, c in cooccur_counter.items()
                if c >= min_df and w not in _STOPWORDS
            }
            top_l2 = [
                w for w, _ in Counter(filtered).most_common(max_l2_per_l1)
            ]

            if not top_l2:
                top_l2 = ["其他"]  # fallback

            # 5. 构建标签节点
            l1_id = f"L1-{idx:02d}"
            self.add(LabelNode(
                id=l1_id,
                name=l1_name,
                level=1,
                description=f"TELEClass 自动构建: {l1_name}",
                keywords=keywords,
            ))

            l2_nodes = []
            for l2_idx, l2_name in enumerate(top_l2, start=1):
                l2_id = f"L2-{idx:02d}-{l2_idx:02d}"
                # L2 关键词 = 共现词本身 + 与种子词组合
                l2_keywords = [l2_name] + [
                    f"{s}{l2_name}" for s in seeds[:2]
                ]
                self.add(LabelNode(
                    id=l2_id,
                    name=l2_name,
                    level=2,
                    parent_id=l1_id,
                    description=f"{l1_name} 下的 {l2_name} 相关反馈",
                    keywords=l2_keywords,
                ))
                l2_nodes.append(l2_name)

            built_log[l1_name] = l2_nodes

        return built_log

    def _invalidate_cache(self) -> None:
        self._children_cache = {}

    def __repr__(self) -> str:
        counts = {i: 0 for i in range(1, 5)}
        for node in self.nodes.values():
            counts[node.level] += 1
        return f"LabelSystem(total={len(self.nodes)}, L1={counts[1]}, L2={counts[2]}, L3={counts[3]}, L4={counts[4]})"


# ── 测试 ──────────────────────────────────────────────────────

def _create_demo_system() -> LabelSystem:
    """创建演示用的标签体系"""
    system = LabelSystem()

    # L1: 品类
    system.add(LabelNode("L1-01", "纸尿裤", 1, description=" diaper 类产品", keywords=["纸尿裤", "尿布", "diaper"]))
    system.add(LabelNode("L1-02", "奶粉", 1, description="婴幼儿奶粉", keywords=["奶粉", "formula", "milk"]))

    # L2: 问题域
    system.add(LabelNode("L2-01", "质量", 2, "L1-01", "产品质量相关问题", ["质量", "品质", "做工", "material"]))
    system.add(LabelNode("L2-02", "物流", 2, "L1-01", "物流配送相关问题", ["物流", "快递", "shipping", "delivery"]))
    system.add(LabelNode("L2-03", "质量", 2, "L1-02", "奶粉质量问题", ["质量", "结块", "异味", "quality"]))

    # L3: 细分类别
    system.add(LabelNode("L3-01", "漏尿问题", 3, "L2-01", "漏尿相关", ["漏", "漏尿", "漏屎", "leak"]))
    system.add(LabelNode("L3-02", "材质舒适度", 3, "L2-01", "材质相关", ["材质", "柔软", "硬", "comfort"]))
    system.add(LabelNode("L3-03", "配送时效", 3, "L2-02", "配送速度", ["慢", "时效", "delay", "late"]))

    # L4: 具体痛点（部分预定义，部分可由进化发现）
    system.add(LabelNode("L4-01", "夜间侧漏", 4, "L3-01", "晚上睡觉时漏", ["夜间", "晚上", "睡觉", "night"]))
    system.add(LabelNode("L4-02", "腰贴太硬", 4, "L3-02", "腰贴/魔术贴过硬", ["腰贴", "魔术贴", "硬", "tab"]))
    system.add(LabelNode("L4-03", "清关延迟", 4, "L3-03", "海关清关慢", ["清关", "海关", "customs", "clearance"]))

    return system


def test_teleclass_bootstrap():
    """测试 TELEClass 最小监督引导"""
    print("\n" + "=" * 60)
    print("测试: TELEClass Bootstrap")
    print("=" * 60)

    # 模拟母婴电商评论语料
    corpus = [
        # 纸尿裤相关
        "纸尿裤晚上总是漏尿，宝宝睡觉不安稳",
        "这款纸尿裤防漏效果很好，整晚都不会漏",
        "宝宝用了红屁股，可能是材质过敏",
        "漏尿问题严重，腰部设计不合理",
        "红屁股反反复复，换了好几个品牌",
        "材质很柔软，但还是会侧漏",
        "夜间漏尿导致床单都湿了",
        "透气性好但吸水一般，容易漏",
        # 奶粉相关
        "奶粉溶解性不好，有结块",
        "宝宝喝完奶粉便秘，排便困难",
        "奶粉味道太腥，宝宝不爱喝",
        "冲泡后有很多泡沫，不知道怎么回事",
        "换了这个奶粉不再便秘了",
        "结块问题很严重，摇不均匀",
        # 其他
        "物流很快，包装完好",
        "客服态度很好，解决问题及时",
        "性价比不错，会回购",
    ] * 5  # 扩充语料量以满足 min_df

    categories = {
        "纸尿裤": ["漏尿", "红屁股"],
        "奶粉": ["结块", "便秘"],
    }

    system = LabelSystem()
    built = system.teleclass_bootstrap(
        categories=categories,
        corpus=corpus,
        max_l2_per_l1=3,
        min_df=3,
    )

    print(f"\n构建结果: {built}")
    print(f"标签体系: {system}")

    # 验证结构
    for l1_name, l2_names in built.items():
        l1_node = system.find_by_name(l1_name)
        assert l1_node is not None, f"L1 节点 {l1_name} 未创建"
        assert l1_node.level == 1
        children = system.get_children(l1_node.id)
        assert len(children) == len(l2_names), f"{l1_name} 子节点数不匹配"
        print(f"\n  {l1_name}(L1) → {[c.name for c in children]}(L2)")

    # 测试关键词匹配
    print("\n--- 关键词匹配测试 ---")
    text = "宝宝用了这个纸尿裤晚上总是漏尿"
    matches = system.match_by_keyword(text)
    print(f"文本: '{text}'")
    print(f"匹配标签: {[(m.name, round(s, 3)) for m, s in matches[:5]]}")

    # 测试带 expander 的场景
    print("\n--- 带 LLM Expander 的引导 ---")

    def mock_expander(category: str, seeds: list[str]) -> list[str]:
        """模拟 LLM 扩展：基于类别返回相关词"""
        expansions = {
            "纸尿裤": ["尿布", "尿不湿", "拉拉裤", "吸水", "透气", "干爽"],
            "奶粉": ["配方", "营养", "蛋白", "乳糖", "DHA", "辅食"],
        }
        return expansions.get(category, [])

    system2 = LabelSystem()
    built2 = system2.teleclass_bootstrap(
        categories=categories,
        corpus=corpus,
        expander=mock_expander,
        max_l2_per_l1=3,
        min_df=3,
    )
    print(f"带扩展器构建结果: {built2}")
    print(f"标签体系: {system2}")

    print("\n" + "=" * 60)
    print("TELEClass Bootstrap 测试通过 ✓")
    print("=" * 60)


def test_label_system():
    """测试标签体系管理"""
    print("=" * 60)
    print("测试: LabelSystem")
    print("=" * 60)

    system = _create_demo_system()
    print(f"\n创建标签体系: {system}")

    # 测试路径查询
    print("\n--- 路径查询 ---")
    path = system.get_path("L4-01")
    print(f"L4-01(夜间侧漏) 的完整路径: {' → '.join(n.name for n in path)}")

    path_names = system.get_path_names("L4-01")
    print(f"路径名称: {path_names}")

    # 测试子标签查询
    print("\n--- 子标签查询 ---")
    children = system.get_children("L2-01")
    print(f"L2-01(质量-纸尿裤) 的子标签: {[c.name for c in children]}")

    # 测试关键词匹配
    print("\n--- 关键词匹配 ---")
    text = "宝宝晚上总是漏尿，纸尿裤质量不好"
    matches = system.match_by_keyword(text)
    print(f"文本: '{text}'")
    print(f"匹配标签: {[(m.name, round(s, 3)) for m, s in matches[:5]]}")

    # 测试叶子节点
    print("\n--- 叶子节点 ---")
    leaves = system.get_leaves()
    print(f"所有叶子标签: {[n.name for n in leaves]}")

    # 测试序列化
    print("\n--- 序列化 ---")
    system.to_json("/tmp/label_system_demo.json")
    restored = LabelSystem.from_json("/tmp/label_system_demo.json")
    print(f"反序列化后: {restored}")
    assert len(restored.nodes) == len(system.nodes)
    print("✓ 序列化/反序列化测试通过")

    print("\n" + "=" * 60)
    print("所有测试通过 ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_label_system()
    test_teleclass_bootstrap()
