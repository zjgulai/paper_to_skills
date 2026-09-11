"""标签进化引擎

实现标签体系的持续自进化:
    - 触发条件判断（时间/数据量/覆盖率）
    - 进化动作执行（新增/合并/淘汰/升级）
    - 进化效果评估
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional

from label_system import LabelNode, LabelSystem


@dataclass
class EvolutionTrigger:
    """进化触发条件"""

    # 时间触发
    time_window_days: int = 30           # 检查周期（天）

    # 数据量触发
    min_new_feedbacks: int = 10000       # 新反馈数量阈值

    # 覆盖率触发
    coverage_threshold: float = 0.85     # 覆盖率低于此值触发进化

    # 候选标签触发
    min_candidate_labels: int = 20       # 未确认候选标签数量阈值


@dataclass
class EvolutionAction:
    """单次进化动作"""

    action_type: str                      # add / merge / retire / promote
    target_id: str                        # 目标标签ID
    details: dict = field(default_factory=dict)
    timestamp: str = ""


class LabelEvolution:
    """标签进化引擎

    监控标签体系健康度，在满足条件时自动执行进化动作。
    所有"删除"类操作均为软删除（标记 dormant），可随时恢复。
    """

    def __init__(
        self,
        label_system: LabelSystem,
        trigger: Optional[EvolutionTrigger] = None,
        min_hit_for_active: int = 50,         # 新标签入库最低命中次数
        similarity_merge_threshold: float = 0.85,  # 合并相似度阈值
        dormant_days: int = 90,               # 休眠判定天数
    ):
        self.label_system = label_system
        self.trigger = trigger or EvolutionTrigger()
        self.min_hit_for_active = min_hit_for_active
        self.similarity_merge_threshold = similarity_merge_threshold
        self.dormant_days = dormant_days

        # 进化历史
        self.history: list[EvolutionAction] = []

        # 统计信息
        self.stats = {
            "total_feedbacks": 0,
            "covered_feedbacks": 0,
            "novel_feedbacks": 0,
            "last_check": "",
        }

    # ── 触发判断 ───────────────────────────────────────────────

    def should_evolve(
        self,
        new_feedbacks_count: int = 0,
        current_coverage: Optional[float] = None,
        candidate_labels_count: int = 0,
    ) -> tuple[bool, str]:
        """判断是否应该触发进化

        Returns:
            (是否触发, 触发原因)
        """
        # 检查数据量
        if new_feedbacks_count >= self.trigger.min_new_feedbacks:
            return True, f"新反馈数量达标: {new_feedbacks_count} >= {self.trigger.min_new_feedbacks}"

        # 检查覆盖率
        if current_coverage is not None and current_coverage < self.trigger.coverage_threshold:
            return True, f"覆盖率过低: {current_coverage:.2%} < {self.trigger.coverage_threshold:.0%}"

        # 检查候选标签数
        if candidate_labels_count >= self.trigger.min_candidate_labels:
            return True, f"候选标签积累: {candidate_labels_count} >= {self.trigger.min_candidate_labels}"

        # 检查时间（简化版：对比 last_check）
        if self.stats["last_check"]:
            last = datetime.fromisoformat(self.stats["last_check"])
            if datetime.now() - last >= timedelta(days=self.trigger.time_window_days):
                return True, f"时间窗口到达: {self.trigger.time_window_days}天"

        return False, ""

    # ── 进化动作 ───────────────────────────────────────────────

    def add_new_label(
        self,
        name: str,
        parent_id: str,
        description: str = "",
        keywords: Optional[list[str]] = None,
    ) -> str:
        """新增标签（从候选提升为正式标签）"""
        # 生成ID
        level = self.label_system.get(parent_id).level + 1
        existing = [n for n in self.label_system.nodes.values() if n.level == level]
        new_id = f"L{level}-{len(existing) + 1:03d}"

        node = LabelNode(
            id=new_id,
            name=name,
            level=level,
            parent_id=parent_id,
            description=description,
            keywords=keywords or [],
            status="active",
            created_at=datetime.now().isoformat(),
        )
        self.label_system.add(node)

        action = EvolutionAction(
            action_type="add",
            target_id=new_id,
            details={"name": name, "parent_id": parent_id},
            timestamp=datetime.now().isoformat(),
        )
        self.history.append(action)

        return new_id

    def merge_labels(self, keep_id: str, merge_id: str) -> None:
        """合并两个相似标签（保留 keep_id，删除 merge_id）"""
        if keep_id not in self.label_system.nodes or merge_id not in self.label_system.nodes:
            raise ValueError("标签不存在")

        keep = self.label_system.nodes[keep_id]
        merge = self.label_system.nodes[merge_id]

        # 合并关键词
        keep.keywords = list(set(keep.keywords + merge.keywords))
        # 合并命中次数
        keep.hit_count += merge.hit_count

        # 迁移子标签
        for child in self.label_system.get_children(merge_id):
            child.parent_id = keep_id
        self.label_system._invalidate_cache()

        # 软删除被合并标签
        merge.status = "dormant"
        merge.description += f" [已合并到 {keep.name}({keep_id})]"

        action = EvolutionAction(
            action_type="merge",
            target_id=merge_id,
            details={"keep_id": keep_id, "keep_name": keep.name},
            timestamp=datetime.now().isoformat(),
        )
        self.history.append(action)

    def retire_label(self, node_id: str) -> None:
        """淘汰休眠标签"""
        node = self.label_system.get(node_id)

        # 检查条件
        if node.status != "active":
            return

        # 标记为休眠
        node.status = "dormant"

        action = EvolutionAction(
            action_type="retire",
            target_id=node_id,
            details={"name": node.name, "reason": "长期无命中"},
            timestamp=datetime.now().isoformat(),
        )
        self.history.append(action)

    def promote_label(self, node_id: str) -> str:
        """将 L4 标签提升为 L3（当子标签过多时）"""
        node = self.label_system.get(node_id)
        if node.level != 4:
            raise ValueError("只有 L4 标签可以提升")

        # 在当前父标签下创建新的 L3
        parent_l3 = self.label_system.get(node.parent_id)
        parent_l2 = self.label_system.get(parent_l3.parent_id)

        # 新 L3 标签
        existing_l3 = [n for n in self.label_system.nodes.values() if n.level == 3]
        new_l3_id = f"L3-{len(existing_l3) + 1:03d}"

        new_l3 = LabelNode(
            id=new_l3_id,
            name=f"{node.name}相关",
            level=3,
            parent_id=parent_l2.id,
            description=f"由 L4 标签 {node.name} 提升而来",
            keywords=node.keywords[:3],
            status="active",
            created_at=datetime.now().isoformat(),
        )
        self.label_system.add(new_l3)

        # 将原 L4 移入新的 L3 下
        node.parent_id = new_l3_id
        self.label_system._invalidate_cache()

        action = EvolutionAction(
            action_type="promote",
            target_id=node_id,
            details={"new_l3_id": new_l3_id, "new_l3_name": new_l3.name},
            timestamp=datetime.now().isoformat(),
        )
        self.history.append(action)

        return new_l3_id

    # ── 批量进化 ───────────────────────────────────────────────

    def run_evolution(
        self,
        candidate_labels: list[dict],
        coverage: float,
        new_feedbacks: int = 0,
    ) -> list[EvolutionAction]:
        """执行一轮完整进化

        Args:
            candidate_labels: 候选标签列表，每个包含 name, parent_name, count
            coverage: 当前覆盖率
            new_feedbacks: 新增反馈数量

        Returns:
            本次执行的所有进化动作
        """
        should, reason = self.should_evolve(new_feedbacks, coverage, len(candidate_labels))
        if not should:
            return []

        print(f"[进化触发] {reason}")

        actions_before = len(self.history)

        # 1. 新增高频候选标签
        for candidate in candidate_labels:
            if candidate.get("count", 0) >= self.min_hit_for_active:
                # 查找父标签ID
                parent = self.label_system.find_by_name(candidate["parent_name"])
                if parent:
                    self.add_new_label(
                        name=candidate["name"].replace("新: ", ""),
                        parent_id=parent.id,
                        description=candidate.get("description", ""),
                        keywords=candidate.get("keywords", []),
                    )

        # 2. 淘汰休眠标签
        cutoff = datetime.now() - timedelta(days=self.dormant_days)
        for node in self.label_system.get_all_active():
            if node.level >= 3 and node.last_hit_at:  # 只对 L3+ 标签做休眠
                last_hit = datetime.fromisoformat(node.last_hit_at)
                if last_hit < cutoff:
                    self.retire_label(node.id)

        # 3. 标签升级检查
        for node in self.label_system.nodes.values():
            if node.level == 3:
                children = self.label_system.get_children(node.id)
                if len(children) >= 10:  # L3 下子标签过多，考虑拆分
                    # 这里简化处理，实际可能需要更复杂的拆分逻辑
                    pass

        self.stats["last_check"] = datetime.now().isoformat()

        return self.history[actions_before:]

    # ── 评估 ───────────────────────────────────────────────────

    def get_coverage(self, predictions: list[dict]) -> float:
        """计算标签覆盖率"""
        if not predictions:
            return 0.0
        covered = sum(1 for p in predictions if not p.get("is_novel", False))
        return covered / len(predictions)

    def get_health_report(self) -> dict:
        """生成标签体系健康报告"""
        total = len(self.label_system.nodes)
        active = len(self.label_system.get_all_active())
        dormant = total - active
        leaves = len(self.label_system.get_leaves())

        by_level = {i: 0 for i in range(1, 5)}
        for node in self.label_system.nodes.values():
            by_level[node.level] += 1

        return {
            "total_labels": total,
            "active": active,
            "dormant": dormant,
            "leaf_labels": leaves,
            "by_level": by_level,
            "evolution_count": len(self.history),
            "last_evolution": self.stats["last_check"],
        }

    def print_health_report(self) -> None:
        """打印健康报告"""
        report = self.get_health_report()
        print("\n" + "=" * 50)
        print("标签体系健康报告")
        print("=" * 50)
        print(f"总标签数: {report['total_labels']} (活跃 {report['active']}, 休眠 {report['dormant']})")
        print(f"叶子标签: {report['leaf_labels']}")
        print(f"层级分布: L1={report['by_level'][1]}, L2={report['by_level'][2]}, L3={report['by_level'][3]}, L4={report['by_level'][4]}")
        print(f"历史进化次数: {report['evolution_count']}")
        print(f"上次检查: {report['last_evolution'] or '从未'}")
        print("=" * 50)


# ── 测试 ──────────────────────────────────────────────────────

def test_evolution():
    """测试标签进化引擎"""
    from label_system import LabelNode, _create_demo_system

    print("=" * 60)
    print("测试: LabelEvolution")
    print("=" * 60)

    system = _create_demo_system()
    trigger = EvolutionTrigger(
        time_window_days=30,
        min_new_feedbacks=100,
        coverage_threshold=0.85,
        min_candidate_labels=2,
    )
    evolution = LabelEvolution(system, trigger, min_hit_for_active=3, dormant_days=30)

    # 测试1: 触发判断
    print("\n--- 触发判断 ---")
    should, reason = evolution.should_evolve(
        new_feedbacks_count=150,
        current_coverage=0.80,
        candidate_labels_count=5,
    )
    print(f"应该进化: {should}, 原因: {reason}")
    assert should is True
    print("✓ 触发判断测试通过")

    # 测试2: 不触发
    should, reason = evolution.should_evolve(
        new_feedbacks_count=50,
        current_coverage=0.90,
        candidate_labels_count=1,
    )
    print(f"应该进化: {should}, 原因: {reason or '未触发'}")
    assert should is False
    print("✓ 不触发测试通过")

    # 测试3: 新增标签
    print("\n--- 新增标签 ---")
    before = len(system.nodes)
    new_id = evolution.add_new_label(
        name="出汗后脱落",
        parent_id="L3-02",
        description="防蚊贴出汗后粘性不足脱落",
        keywords=["出汗", "脱落", "粘不住"],
    )
    print(f"新增标签: {new_id} = {system.get(new_id).name}")
    assert len(system.nodes) == before + 1
    print("✓ 新增标签测试通过")

    # 测试4: 合并标签
    print("\n--- 合并标签 ---")
    # 先创建一个将被合并的标签
    system.add(LabelNode(
        id="L4-04", name="晚上漏尿", level=4, parent_id="L3-01",
        keywords=["晚上", "漏尿", "night"], status="active"
    ))
    evolution.merge_labels("L4-01", "L4-04")
    assert system.get("L4-04").status == "dormant"
    print("✓ 合并标签测试通过")

    # 测试5: 淘汰标签
    print("\n--- 淘汰标签 ---")
    # 模拟一个长期无命中的标签
    old_node = LabelNode(
        id="L4-99", name="过时标签", level=4, parent_id="L3-01",
        last_hit_at=(datetime.now() - timedelta(days=100)).isoformat(),
    )
    system.add(old_node)
    evolution.retire_label("L4-99")
    assert system.get("L4-99").status == "dormant"
    print("✓ 淘汰标签测试通过")

    # 测试6: 批量进化
    print("\n--- 批量进化 ---")
    candidates = [
        {"name": "新: 腰贴过敏", "parent_name": "质量", "count": 5, "description": "腰贴部位过敏"},
        {"name": "新: 尺码偏小", "parent_name": "质量", "count": 2, "description": "尺码偏小"},
    ]
    actions = evolution.run_evolution(candidates, coverage=0.80, new_feedbacks=150)
    print(f"执行了 {len(actions)} 个进化动作:")
    for a in actions:
        print(f"  [{a.action_type}] {a.target_id}")

    # 测试7: 健康报告
    print("\n--- 健康报告 ---")
    evolution.print_health_report()

    print("\n" + "=" * 60)
    print("进化引擎测试完成 ✓")
    print("=" * 60)


if __name__ == "__main__":
    test_evolution()
