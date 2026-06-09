"""
Guardrailed CATE-NBA: 带护栏的增量下一最佳行动优化框架
==========================================================

论文来源: Guardrailed Uplift Targeting: A Causal Optimization Playbook for Marketing Strategy
arXiv ID: 2512.19805 (2026年2月修订)

三层架构:
  Layer 1 - CATE 因果估算层:  估计每个用户对每种行动的条件平均处理效应
  Layer 2 - 护栏约束层:       把业务规则写成数学约束（防食人化 / 预算 / 体验红线）
  Layer 3 - 约束分配规划层:   带约束的多维背包 → 整数规划或贪心求解，输出"干仗名单"

使用方法:
  python model.py          # 运行完整 demo + 自检
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class Action:
    """单种营销行动定义"""
    action_id: str         # 行动标识，如 "coupon_50", "free_sample"
    name: str              # 显示名称
    unit_cost: float       # 每次触达成本（元 / 美元）
    max_sends: int = 1     # 每用户最多发送次数（默认 1，即互斥约束）


@dataclass
class GuardrailConfig:
    """护栏参数配置"""
    # 防食人化（Cannibalization）阈值: CATE > threshold 且 base_prob > high_value_threshold 的用户不发大额券
    cannibalization_base_prob_threshold: float = 0.7   # 基础转化概率上限
    cannibalization_cate_discount: float = 0.5         # 食人化用户的 CATE 打折系数

    # 预算上限（总成本约束，单位与 Action.unit_cost 一致）
    total_budget: float = 10_000.0

    # 用户体验约束: 每用户最多被打扰 N 次（跨所有行动）
    max_actions_per_user: int = 1

    # 最低增量门槛: CATE < min_cate 的 (user, action) 对直接过滤，不进入规划器
    min_cate: float = 0.01


@dataclass
class AllocationResult:
    """分配结果"""
    assignments: pd.DataFrame   # 列: user_id, action_id, cate, cost
    total_cost: float
    total_expected_uplift: float
    summary: Dict[str, int]     # action_id → 分配人数


# ---------------------------------------------------------------------------
# Layer 1: Mock CATE 估算（模拟 Causal Forest / Meta-learner 输出）
# ---------------------------------------------------------------------------

class MockCATEEstimator:
    """
    模拟 CATE 估算器（实际业务中替换为 causalml / econml 的真实模型）

    CATE(user, action) = P(convert | T=action) - P(convert | T=None)
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def fit(self, X: pd.DataFrame) -> "MockCATEEstimator":
        """占位 fit，实际替换为 CausalForest.fit()"""
        self.n_features = X.shape[1]
        return self

    def predict_cate(
        self,
        X: pd.DataFrame,
        actions: List[Action],
    ) -> pd.DataFrame:
        """
        返回 CATE 矩阵: shape (n_users, n_actions)
        列名为 action.action_id
        """
        n = len(X)
        cate_dict: Dict[str, np.ndarray] = {}

        for action in actions:
            # Mock: 以用户特征简单线性函数 + 高斯噪声模拟 CATE
            # 实际替换为: model.effect(X, T0=0, T1=action_idx)
            base_signal = 0.05 + 0.15 * X["recency_score"].values
            cost_discount = 0.8 if action.unit_cost > 30 else 1.0
            noise = self.rng.normal(0, 0.03, size=n)
            cate_dict[action.action_id] = np.clip(
                base_signal * cost_discount + noise, -0.1, 0.5
            )

        return pd.DataFrame(cate_dict, index=X.index)

    def predict_base_prob(self, X: pd.DataFrame) -> np.ndarray:
        """预测无干预下用户自然转化概率 P(convert | T=None)"""
        # Mock: RFM 综合得分简单映射
        prob = 0.1 + 0.6 * X["rfm_score"].values + self.rng.normal(0, 0.02, len(X))
        return np.clip(prob, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Layer 2: 护栏过滤（Guardrail Filter）
# ---------------------------------------------------------------------------

class GuardrailFilter:
    """
    把业务规则转化为约束，过滤或打折不符合规则的 (user, action) 对

    规则 1 - 防食人化: 基础概率已经很高的用户发大额券会浪费预算
    规则 2 - CATE 最低门槛: 增量不显著的配对直接丢弃
    """

    def __init__(self, config: GuardrailConfig):
        self.cfg = config

    def apply(
        self,
        cate_matrix: pd.DataFrame,
        base_prob: np.ndarray,
        actions: List[Action],
    ) -> pd.DataFrame:
        """
        返回经过护栏处理的 effective_cate 矩阵（shape 与 cate_matrix 相同）
        对不符合规则的 (user, action) 对，将 effective_cate 设为 0 或打折
        """
        effective = cate_matrix.copy()

        high_value_mask = base_prob >= self.cfg.cannibalization_base_prob_threshold

        for action in actions:
            col = action.action_id
            if col not in effective.columns:
                continue
            # 规则 1: 食人化惩罚 — 高净值用户发高额券打折
            if action.unit_cost > 20:  # 高额券阈值
                effective.loc[high_value_mask, col] *= self.cfg.cannibalization_cate_discount
                logger.info(
                    "规则1-食人化惩罚: 对 %d 个高净值用户的 %s CATE 打折 %.0f%%",
                    int(high_value_mask.sum()),
                    col,
                    (1 - self.cfg.cannibalization_cate_discount) * 100,
                )

            # 规则 2: 最低增量门槛过滤
            below_min = effective[col] < self.cfg.min_cate
            effective.loc[below_min, col] = 0.0

        logger.info(
            "护栏后有效配对数: %d / %d",
            int((effective > 0).values.sum()),
            effective.size,
        )
        return effective


# ---------------------------------------------------------------------------
# Layer 3: 约束分配规划（Constrained Allocation / Greedy Knapsack）
# ---------------------------------------------------------------------------

class GreedyKnapsackPlanner:
    """
    带预算 + 用户体验约束的贪心背包分配器

    目标: maximize Σ cate(u, a) * x(u, a)
    约束:
      Σ_{u,a} cost(a) * x(u,a) ≤ total_budget       (预算)
      Σ_a x(u,a) ≤ max_actions_per_user  ∀u           (体验)
      x(u,a) ∈ {0,1}                                   (二元分配)

    贪心策略: 按 CATE / cost (效价比) 降序排列所有可行配对, 逐一分配
    (中小规模实际可替换为 PuLP/ortools 的 MIP 精确求解)
    """

    def __init__(self, config: GuardrailConfig):
        self.cfg = config

    def solve(
        self,
        effective_cate: pd.DataFrame,
        actions: List[Action],
    ) -> AllocationResult:
        """
        输入: effective_cate — 护栏后的 CATE 矩阵 (n_users × n_actions)
        输出: AllocationResult
        """
        action_map: Dict[str, Action] = {a.action_id: a for a in actions}

        # 构建候选 (user_id, action_id, cate, cost, efficiency) 列表
        rows = []
        for action_id, col in effective_cate.items():
            act = action_map[action_id]
            for user_id, cate_val in col.items():
                if cate_val <= 0:
                    continue
                efficiency = cate_val / max(act.unit_cost, 1e-6)
                rows.append({
                    "user_id": user_id,
                    "action_id": action_id,
                    "cate": cate_val,
                    "cost": act.unit_cost,
                    "efficiency": efficiency,
                })

        if not rows:
            logger.warning("没有有效候选配对，分配结果为空")
            return AllocationResult(
                assignments=pd.DataFrame(),
                total_cost=0.0,
                total_expected_uplift=0.0,
                summary={},
            )

        candidates = pd.DataFrame(rows).sort_values("efficiency", ascending=False)

        # 贪心分配
        remaining_budget = self.cfg.total_budget
        user_action_count: Dict[str, int] = {}
        selected: List[Dict] = []

        for _, row in candidates.iterrows():
            uid = row["user_id"]
            aid = row["action_id"]

            # 检查用户体验约束
            if user_action_count.get(uid, 0) >= self.cfg.max_actions_per_user:
                continue
            # 检查预算约束
            if row["cost"] > remaining_budget:
                continue

            selected.append({
                "user_id": uid,
                "action_id": aid,
                "cate": row["cate"],
                "cost": row["cost"],
            })
            remaining_budget -= row["cost"]
            user_action_count[uid] = user_action_count.get(uid, 0) + 1

        assignments = pd.DataFrame(selected)

        if assignments.empty:
            summary: Dict[str, int] = {}
            total_cost = 0.0
            total_uplift = 0.0
        else:
            summary = assignments.groupby("action_id")["user_id"].count().to_dict()
            total_cost = assignments["cost"].sum()
            total_uplift = assignments["cate"].sum()

        logger.info(
            "分配完成 | 触达用户: %d | 总成本: %.1f | 预期增量: %.3f",
            len(assignments),
            total_cost,
            total_uplift,
        )

        return AllocationResult(
            assignments=assignments,
            total_cost=total_cost,
            total_expected_uplift=total_uplift,
            summary=summary,
        )


# ---------------------------------------------------------------------------
# 完整 Pipeline
# ---------------------------------------------------------------------------

class GuardrailedCATENBA:
    """
    Guardrailed CATE-NBA 完整流水线

    用法:
        model = GuardrailedCATENBA(actions=actions, config=guardrail_cfg)
        result = model.run(user_features_df)
        print(result.summary)
    """

    def __init__(
        self,
        actions: List[Action],
        config: Optional[GuardrailConfig] = None,
        estimator: Optional[MockCATEEstimator] = None,
    ):
        self.actions = actions
        self.config = config or GuardrailConfig()
        self.estimator = estimator or MockCATEEstimator()
        self._guardrail = GuardrailFilter(self.config)
        self._planner = GreedyKnapsackPlanner(self.config)

    def run(self, X: pd.DataFrame) -> AllocationResult:
        """
        端到端运行:
          X: 用户特征 DataFrame，至少包含 recency_score [0,1] 和 rfm_score [0,1]
        """
        logger.info("=== Layer 1: CATE 估算 ===")
        self.estimator.fit(X)
        cate_matrix = self.estimator.predict_cate(X, self.actions)
        base_prob = self.estimator.predict_base_prob(X)
        logger.info("CATE 矩阵 shape: %s", cate_matrix.shape)

        logger.info("=== Layer 2: 护栏过滤 ===")
        effective_cate = self._guardrail.apply(cate_matrix, base_prob, self.actions)

        logger.info("=== Layer 3: 约束分配 ===")
        result = self._planner.solve(effective_cate, self.actions)

        return result


# ---------------------------------------------------------------------------
# 辅助: 生成 Mock 用户数据
# ---------------------------------------------------------------------------

def generate_mock_users(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """生成模拟用户特征 DataFrame"""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "user_id": [f"U{i:05d}" for i in range(n)],
        "recency_score": rng.uniform(0, 1, n),      # 近期活跃得分
        "rfm_score": rng.uniform(0, 1, n),           # RFM 综合得分
        "days_since_last_order": rng.integers(1, 180, n),
        "historical_orders": rng.integers(0, 20, n),
    }).set_index("user_id")
    return df


# ---------------------------------------------------------------------------
# 自检测试
# ---------------------------------------------------------------------------

def _test_basic_flow() -> None:
    """测试 1: 基础流程能够正常完成并返回正确类型"""
    logger.info("\n>>> 测试 1: 基础流程")

    actions = [
        Action("coupon_10", "满减券10元", unit_cost=10.0),
        Action("coupon_50", "满减券50元", unit_cost=50.0),
        Action("free_sample", "免费小样", unit_cost=5.0),
    ]
    cfg = GuardrailConfig(total_budget=5_000.0, min_cate=0.02)
    X = generate_mock_users(n=200)

    model = GuardrailedCATENBA(actions=actions, config=cfg)
    result = model.run(X)

    assert isinstance(result, AllocationResult), "结果类型错误"
    assert isinstance(result.assignments, pd.DataFrame), "assignments 类型错误"
    assert result.total_cost >= 0, "总成本不能为负"
    assert result.total_expected_uplift >= 0, "总增量不能为负"
    logger.info("测试 1 通过 ✓ | 分配用户数: %d", len(result.assignments))


def _test_budget_constraint() -> None:
    """测试 2: 预算约束 — 总成本不超过预算"""
    logger.info("\n>>> 测试 2: 预算约束")

    actions = [Action("coupon_100", "大额券100", unit_cost=100.0)]
    cfg = GuardrailConfig(total_budget=500.0)
    X = generate_mock_users(n=100)

    model = GuardrailedCATENBA(actions=actions, config=cfg)
    result = model.run(X)

    assert result.total_cost <= cfg.total_budget + 1e-6, (
        f"超预算! 总成本={result.total_cost:.1f} > 预算={cfg.total_budget}"
    )
    logger.info("测试 2 通过 ✓ | 总成本: %.1f ≤ 预算: %.1f", result.total_cost, cfg.total_budget)


def _test_user_experience_constraint() -> None:
    """测试 3: 用户体验约束 — 每个用户最多被分配 max_actions_per_user 次"""
    logger.info("\n>>> 测试 3: 用户体验约束")

    actions = [
        Action("act_a", "行动A", unit_cost=1.0),
        Action("act_b", "行动B", unit_cost=1.0),
        Action("act_c", "行动C", unit_cost=1.0),
    ]
    cfg = GuardrailConfig(total_budget=100_000.0, max_actions_per_user=1, min_cate=0.0)
    X = generate_mock_users(n=50)

    model = GuardrailedCATENBA(actions=actions, config=cfg)
    result = model.run(X)

    if not result.assignments.empty:
        per_user = result.assignments.groupby("user_id")["action_id"].count()
        max_per_user = per_user.max()
        assert max_per_user <= cfg.max_actions_per_user, (
            f"用户体验违规! 最多被分配 {max_per_user} 次 > 限制 {cfg.max_actions_per_user}"
        )
    logger.info("测试 3 通过 ✓ | 每用户最多行动次数: %d", cfg.max_actions_per_user)


def _test_cannibalization_guardrail() -> None:
    """测试 4: 防食人化护栏 — 高净值用户的高额券 CATE 应被打折"""
    logger.info("\n>>> 测试 4: 防食人化护栏")

    rng = np.random.default_rng(0)
    # 构造全部为高净值用户（rfm_score ≈ 1.0）
    n = 50
    X = pd.DataFrame({
        "recency_score": rng.uniform(0.8, 1.0, n),
        "rfm_score": rng.uniform(0.8, 1.0, n),   # 高净值
        "days_since_last_order": rng.integers(1, 30, n),
        "historical_orders": rng.integers(10, 20, n),
    }, index=[f"HV{i:03d}" for i in range(n)])

    actions = [Action("coupon_50", "大额券50元", unit_cost=50.0)]
    cfg = GuardrailConfig(
        cannibalization_base_prob_threshold=0.5,
        cannibalization_cate_discount=0.5,
        total_budget=100_000.0,
    )

    estimator = MockCATEEstimator(seed=0)
    guardrail = GuardrailFilter(cfg)
    estimator.fit(X)
    cate_raw = estimator.predict_cate(X, actions)
    base_prob = estimator.predict_base_prob(X)
    cate_after = guardrail.apply(cate_raw, base_prob, actions)

    # 高净值用户的 effective CATE 应 ≤ 原始 CATE
    high_value = base_prob >= cfg.cannibalization_base_prob_threshold
    if high_value.any():
        assert (cate_after.loc[high_value, "coupon_50"].values
                <= cate_raw.loc[high_value, "coupon_50"].values + 1e-9).all(), \
            "防食人化护栏未生效"
    logger.info("测试 4 通过 ✓ | 高净值用户数: %d", int(high_value.sum()))


def _test_empty_budget() -> None:
    """测试 5: 极端情况 — 预算为 0 时分配结果为空"""
    logger.info("\n>>> 测试 5: 极端情况 - 零预算")

    actions = [Action("coupon_10", "券10", unit_cost=10.0)]
    cfg = GuardrailConfig(total_budget=0.0)
    X = generate_mock_users(n=30)

    model = GuardrailedCATENBA(actions=actions, config=cfg)
    result = model.run(X)

    assert result.total_cost == 0.0, "零预算下不应有成本"
    assert len(result.assignments) == 0, "零预算下不应有分配"
    logger.info("测试 5 通过 ✓")


def _test_large_scale_performance() -> None:
    """测试 6: 大规模场景 — 10万用户 + 5个行动"""
    import time
    logger.info("\n>>> 测试 6: 大规模性能测试 (n=100,000)")

    actions = [
        Action("coupon_5",   "券5元",   unit_cost=5.0),
        Action("coupon_20",  "券20元",  unit_cost=20.0),
        Action("coupon_50",  "券50元",  unit_cost=50.0),
        Action("free_gift",  "赠品",    unit_cost=8.0),
        Action("sms_remind", "短信提醒", unit_cost=0.5),
    ]
    cfg = GuardrailConfig(total_budget=50_000.0, min_cate=0.03)
    X = generate_mock_users(n=100_000)

    model = GuardrailedCATENBA(actions=actions, config=cfg)
    start = time.time()
    result = model.run(X)
    elapsed = time.time() - start

    assert result.total_cost <= cfg.total_budget + 1e-6, "超预算"
    logger.info(
        "测试 6 通过 ✓ | 耗时: %.2fs | 触达用户: %d | 总成本: %.0f",
        elapsed, len(result.assignments), result.total_cost,
    )


def _print_demo_report(result: AllocationResult) -> None:
    """打印 Demo 报告"""
    print("\n" + "=" * 60)
    print("📊 Guardrailed CATE-NBA 分配报告")
    print("=" * 60)
    print(f"总触达用户数:  {len(result.assignments):,}")
    print(f"总营销成本:    {result.total_cost:,.1f} 元")
    print(f"预期总增量:    {result.total_expected_uplift:.3f}")
    if result.total_cost > 0:
        print(f"增量/成本比:   {result.total_expected_uplift / result.total_cost:.5f}")
    print("\n行动分配明细:")
    for action_id, count in sorted(result.summary.items(), key=lambda x: -x[1]):
        print(f"  {action_id:20s}: {count:,} 人")
    if not result.assignments.empty:
        print("\n前5条分配记录:")
        print(result.assignments.head(5).to_string(index=False))
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Demo 入口 + 自检"""
    print("🚀 Guardrailed CATE-NBA - 跑批演示\n")

    # --- Demo 场景: 跨境电商沉默用户精准促活 ---
    actions = [
        Action("coupon_20",  "满减券20元",  unit_cost=20.0),
        Action("coupon_50",  "满减券50元",  unit_cost=50.0),
        Action("free_sample", "免费小样",   unit_cost=8.0),
    ]

    cfg = GuardrailConfig(
        cannibalization_base_prob_threshold=0.70,
        cannibalization_cate_discount=0.50,
        total_budget=10_000.0,
        max_actions_per_user=1,
        min_cate=0.02,
    )

    X = generate_mock_users(n=5_000)
    model = GuardrailedCATENBA(actions=actions, config=cfg)
    result = model.run(X)
    _print_demo_report(result)

    # --- 自检 ---
    print("\n🔬 运行自检套件...")
    _test_basic_flow()
    _test_budget_constraint()
    _test_user_experience_constraint()
    _test_cannibalization_guardrail()
    _test_empty_budget()
    _test_large_scale_performance()

    print("\n✅ 所有自检通过！Guardrailed CATE-NBA 模块验证完毕。")


if __name__ == "__main__":
    main()
