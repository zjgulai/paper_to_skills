"""
MAB 实时决策服务
集成数据加载和模型推理
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import AdDataLoader
from ab_testing.multi_armed_bandit import ThompsonSampling
import json
import os
from datetime import datetime


class MABDecisionService:
    """MAB 决策服务"""

    def __init__(self, min_exploration_samples: int = 100):
        self.ad_loader = AdDataLoader()
        self.bandit = None
        self.min_exploration_samples = min_exploration_samples
        self.initialized = False

    def initialize(self, creatives: list = None):
        """初始化 MAB 模型"""
        if creatives is None:
            creatives = self.ad_loader.get_creatives()

        self.bandit = ThompsonSampling(
            n_arms=len(creatives),
            arm_names=creatives
        )
        self.initialized = True

        # 从历史数据训练
        self._train_from_history()

        print(f"MAB 服务初始化完成，素材: {creatives}")
        return self

    def _train_from_history(self):
        """从历史数据训练"""
        df = self.ad_loader.load_daily_stats()

        for _, row in df.iterrows():
            creative = row['creative_id']
            if creative not in self.bandit.arm_names:
                continue
            arm_idx = self.bandit.arm_names.index(creative)

            # 更新点击
            clicks = int(row['clicks'])
            for _ in range(clicks):
                self.bandit.update(arm_idx, reward=1)

            # 更新曝光（未点击）
            impressions = int(row['impressions'])
            for _ in range(impressions - clicks):
                self.bandit.update(arm_idx, reward=0)

    def select_creative(self) -> str:
        """选择最佳素材"""
        if not self.initialized:
            self.initialize()

        selected_idx = self.bandit.select_arm()
        return self.bandit.arm_names[selected_idx]

    def update_feedback(self, creative: str, reward: int):
        """更新反馈"""
        arm_idx = self.bandit.arm_names.index(creative)
        self.bandit.update(arm_idx, reward)

    def get_recommendation(self) -> dict:
        """获取当前推荐"""
        dist = self.bandit.get_distribution()

        sorted_creatives = sorted(
            dist.items(),
            key=lambda x: x[1]['weight'],
            reverse=True
        )

        return {
            'recommended': sorted_creatives[0][0],
            'alternatives': [c[0] for c in sorted_creatives[1:]],
            'distribution': dist,
            'timestamp': datetime.now().isoformat()
        }

    def should_explore(self) -> bool:
        """判断是否应该探索"""
        total = sum(self.bandit.counts)
        return total < self.min_exploration_samples * len(self.bandit.arm_names)


def main():
    print("=" * 60)
    print("MAB 决策服务测试")
    print("=" * 60)

    # 初始化服务
    print("\n[1] 初始化服务...")
    service = MABDecisionService()
    service.initialize()

    # 获取推荐
    print("\n[2] 获取推荐...")
    recommendation = service.get_recommendation()
    print(f"   推荐素材: {recommendation['recommended']}")
    print(f"   流量分配:")
    for name, info in recommendation['distribution'].items():
        print(f"     {name}: {info['weight']*100:.1f}%")

    # 模拟选择
    print("\n[3] 模拟素材选择...")
    selected = service.select_creative()
    print(f"   选中素材: {selected}")

    # 模拟反馈
    print("\n[4] 模拟反馈...")
    service.update_feedback(selected, reward=1)
    print(f"   更新完成: {selected} -> 转化")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()