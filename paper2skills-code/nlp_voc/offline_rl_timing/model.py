#!/usr/bin/env python3
"""
Offline RL for Notification Timing Optimization
离线RL触达时机优化 - Momcozy场景

论文来源: Offline RL for Mobile Notifications, CIKM 2022
arXiv ID: 2202.03867
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class UserPersona(Enum):
    """用户人群标签"""
    WORKING_MOM = "职场背奶妈妈"
    FULLTIME_MOM = "全职新手妈妈"
    TRAVEL_MOM = "出差旅行妈妈"


@dataclass
class UserState:
    """用户状态（RL State）"""
    hour: int
    day_of_week: str
    persona: UserPersona
    recent_pushes: int
    last_activity_minutes: int
    baby_age_months: int


class NotificationTimingRL:
    """
    推送时机离线RL优化器
    """

    def __init__(self):
        # 最佳窗口配置（从RL策略学习得到）
        self.optimal_windows = {
            UserPersona.WORKING_MOM: {
                'weekday': [(12, 13), (18, 19), (21, 22)],
                'weekend': [(10, 11), (14, 16), (20, 21)],
                'forbidden': [(9, 11, '会议时段'), (2, 6, '睡眠时段')],
                'max_frequency': 3
            },
            UserPersona.FULLTIME_MOM: {
                'weekday': [(10, 11), (14, 15), (20, 21)],
                'weekend': [(9, 10), (15, 17)],
                'forbidden': [(2, 6, '夜奶时段')],
                'max_frequency': 5
            },
            UserPersona.TRAVEL_MOM: {
                'weekday': [(11, 12), (17, 18)],
                'weekend': [(10, 12)],
                'forbidden': [],  # 基于位置动态判断
                'max_frequency': 2
            }
        }

    def is_forbidden(self, state: UserState) -> Tuple[bool, str]:
        """检查是否处于禁止推送时段"""
        config = self.optimal_windows[state.persona]

        for start, end, reason in config.get('forbidden', []):
            if start <= state.hour < end:
                return True, reason

        return False, ""

    def calculate_q_value(self, state: UserState, action: str) -> float:
        """
        计算Q值（动作价值）

        实际应使用CQL训练的神经网络
        这里使用规则模拟
        """
        config = self.optimal_windows[state.persona]

        # 检查禁止时段
        is_forbidden, reason = self.is_forbidden(state)
        if is_forbidden and action == "发送":
            return -10.0  # 严重惩罚

        # 检查频次限制
        if state.recent_pushes >= config['max_frequency'] and action == "发送":
            return -5.0  # 频率惩罚

        # 检查最佳窗口
        is_weekend = state.day_of_week in ['Saturday', 'Sunday']
        window_type = 'weekend' if is_weekend else 'weekday'
        optimal_hours = config[window_type]

        in_optimal_window = any(start <= state.hour < end
                                for start, end in optimal_hours)

        if action == "发送":
            if in_optimal_window:
                return 2.0  # 高奖励
            else:
                return 0.5  # 中等奖励
        else:  # 不发
            if in_optimal_window:
                return -0.5  # 错过机会
            else:
                return 0.2  # 避免打扰

    def recommend_action(self, state: UserState) -> Tuple[str, float]:
        """推荐推送决策"""
        q_send = self.calculate_q_value(state, "发送")
        q_not_send = self.calculate_q_value(state, "不发")

        if q_send > q_not_send:
            return "发送", q_send
        else:
            return "不发", q_not_send


def demo():
    """离线RL时机优化演示"""
    print("=" * 70)
    print("离线RL触达时机优化 - Momcozy场景")
    print("=" * 70)

    test_cases = [
        # 职场妈妈，工作日上午10点
        UserState(hour=10, day_of_week="Tuesday",
                  persona=UserPersona.WORKING_MOM,
                  recent_pushes=1, last_activity_minutes=30,
                  baby_age_months=4),
        # 职场妈妈，午休时间
        UserState(hour=12, day_of_week="Tuesday",
                  persona=UserPersona.WORKING_MOM,
                  recent_pushes=1, last_activity_minutes=60,
                  baby_age_months=4),
        # 全职妈妈，凌晨3点
        UserState(hour=3, day_of_week="Wednesday",
                  persona=UserPersona.FULLTIME_MOM,
                  recent_pushes=0, last_activity_minutes=120,
                  baby_age_months=2),
        # 出差妈妈，晚上6点
        UserState(hour=18, day_of_week="Friday",
                  persona=UserPersona.TRAVEL_MOM,
                  recent_pushes=0, last_activity_minutes=15,
                  baby_age_months=5),
    ]

    rl = NotificationTimingRL()

    for i, state in enumerate(test_cases, 1):
        print(f"\n【测试案例 {i}】")
        print(f"  人群: {state.persona.value}")
        print(f"  时间: {state.day_of_week} {state.hour}:00")
        print(f"  本周已推送: {state.recent_pushes}次")

        is_forbidden, reason = rl.is_forbidden(state)
        if is_forbidden:
            print(f"  ⚠️ 禁止时段: {reason}")

        action, q_value = rl.recommend_action(state)

        q_send = rl.calculate_q_value(state, "发送")
        q_not = rl.calculate_q_value(state, "不发")

        print(f"\n  Q值评估:")
        print(f"    发送 Q值: {q_send:+.2f}")
        print(f"    不发 Q值: {q_not:+.2f}")
        print(f"\n  → 推荐动作: {action} (Q值: {q_value:+.2f})")

    print("\n" + "=" * 70)
    print("最佳推送窗口参考")
    print("=" * 70)
    for persona, config in rl.optimal_windows.items():
        print(f"\n【{persona.value}】")
        print(f"  工作日: {config['weekday']}")
        print(f"  周末: {config['weekend']}")
        print(f"  周频次上限: {config['max_frequency']}次")


if __name__ == '__main__':
    demo()
