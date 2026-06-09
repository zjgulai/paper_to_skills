"""
Thompson Sampling MAB — 汤普森采样多臂老虎机
paper2skills-code: 02-A_B实验 | 母婴出海跨境电商
"""
from __future__ import annotations
import random, math
from dataclasses import dataclass, field


@dataclass
class Arm:
    arm_id: str
    true_cvr: float       # 真实转化率（仿真用）
    alpha: float = 1.0    # Beta 分布参数（成功次数 + 1）
    beta: float = 1.0     # Beta 分布参数（失败次数 + 1）
    impressions: int = 0
    conversions: int = 0

    @property
    def observed_cvr(self) -> float:
        return self.conversions / max(self.impressions, 1)

    def sample(self) -> float:
        """从 Beta(alpha, beta) 采样"""
        return random.betavariate(self.alpha, self.beta)

    def update(self, reward: int) -> None:
        self.impressions += 1
        self.conversions += reward
        self.alpha += reward
        self.beta += (1 - reward)


class ThompsonSamplingMAB:
    """汤普森采样：自适应分配流量给最优 Arm"""
    def __init__(self, arms: list[Arm]):
        self.arms = arms

    def select(self) -> Arm:
        samples = [(arm.sample(), arm) for arm in self.arms]
        return max(samples, key=lambda x: x[0])[1]

    def step(self) -> tuple[Arm, int]:
        arm = self.select()
        reward = 1 if random.random() < arm.true_cvr else 0
        arm.update(reward)
        return arm, reward

    def run(self, n_rounds: int, seed: int = 42) -> dict:
        random.seed(seed)
        history = {a.arm_id: [] for a in self.arms}
        total_reward = 0
        for _ in range(n_rounds):
            arm, reward = self.step()
            history[arm.arm_id].append(reward)
            total_reward += reward
        best = max(self.arms, key=lambda a: a.true_cvr)
        regret = best.true_cvr * n_rounds - total_reward
        return {"total_reward": total_reward, "cumulative_regret": round(regret, 1),
                "arm_stats": {a.arm_id: {"impressions": a.impressions,
                                          "conversions": a.conversions,
                                          "observed_cvr": round(a.observed_cvr, 4)}
                              for a in self.arms}}


def run_thompson_demo():
    arms = [
        Arm("listing_v1", true_cvr=0.030),
        Arm("listing_v2", true_cvr=0.038),
        Arm("listing_v3", true_cvr=0.025),
    ]
    mab = ThompsonSamplingMAB(arms)
    results = mab.run(n_rounds=10000)

    print("=== Thompson Sampling MAB（Listing 图片优化）===")
    for arm_id, stats in results["arm_stats"].items():
        arm = next(a for a in arms if a.arm_id == arm_id)
        print(f"  {arm_id}: 真实 CVR={arm.true_cvr:.3f} | "
              f"展示={stats['impressions']:,} | 观测 CVR={stats['observed_cvr']:.4f}")

    best = max(arms, key=lambda a: a.impressions)
    print(f"自动倾斜到最优 Arm: {best.arm_id} (展示 {best.impressions:,} 次)")
    print(f"累计 Regret: {results['cumulative_regret']:.0f}")
    print("✅ Thompson Sampling 演示完成")
if __name__ == "__main__":
    run_thompson_demo()
