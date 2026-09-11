"""
STAN: Stage-Adaptive Network for User Lifecycle Modeling
适用于母婴出海电商AIPL标签体系构建
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


@dataclass
class UserBehavior:
    """用户行为数据单元"""
    user_id: str
    event_type: str  # 'click', 'cart', 'purchase', 'review'
    category_id: int
    baby_stage: str  # '0-6m', '6-12m', '1-3y', '3y+'
    timestamp: pd.Timestamp
    value: float = 0.0  # 订单金额或点击权重


class LifecycleStageEncoder(nn.Module):
    """
    生命周期阶段编码器
    通过用户行为序列学习阶段表示
    """
    def __init__(self, num_categories: int, num_stages: int = 4, embed_dim: int = 64, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_stages = num_stages
        self.category_embed = nn.Embedding(num_categories, embed_dim // 2)
        self.event_embed = nn.Embedding(4, embed_dim // 4)
        self.baby_stage_embed = nn.Embedding(4, embed_dim // 4)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.stage_classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_stages)
        )

    def forward(self, behaviors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = behaviors.shape[:2]
        cat_emb = self.category_embed(behaviors[:, :, 0])
        evt_emb = self.event_embed(behaviors[:, :, 1])
        stage_emb = self.baby_stage_embed(behaviors[:, :, 2])
        x = torch.cat([cat_emb, evt_emb, stage_emb], dim=-1)
        attn_out, _ = self.attention(x, x, x)
        lifecycle_emb = attn_out.mean(dim=1)
        stage_logits = self.stage_classifier(lifecycle_emb)
        return stage_logits, lifecycle_emb


class TaskAdaptiveHead(nn.Module):
    """任务自适应头 - 根据生命周期阶段动态调整多任务权重"""
    def __init__(self, embed_dim: int, num_tasks: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_weight_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tasks),
            nn.Softmax(dim=-1)
        )
        self.task_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
            for _ in range(num_tasks)
        ])

    def forward(self, lifecycle_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        task_weights = self.task_weight_net(lifecycle_emb)
        task_outputs = torch.cat([head(lifecycle_emb) for head in self.task_heads], dim=-1)
        return task_outputs, task_weights


class STANLifecycleModel(nn.Module):
    """STAN完整模型 - 整合生命周期编码和任务自适应预测"""
    def __init__(self, num_categories: int, num_stages: int = 4, embed_dim: int = 64, hidden_dim: int = 128, num_tasks: int = 3):
        super().__init__()
        self.lifecycle_encoder = LifecycleStageEncoder(num_categories, num_stages, embed_dim, hidden_dim)
        self.task_adaptive_head = TaskAdaptiveHead(embed_dim, num_tasks)

    def forward(self, behaviors: torch.Tensor) -> Dict[str, torch.Tensor]:
        stage_logits, lifecycle_emb = self.lifecycle_encoder(behaviors)
        task_outputs, task_weights = self.task_adaptive_head(lifecycle_emb)
        stage_probs = F.softmax(stage_logits, dim=-1)
        predicted_stage = torch.argmax(stage_probs, dim=-1)
        return {
            'stage_logits': stage_logits,
            'stage_probs': stage_probs,
            'predicted_stage': predicted_stage,
            'task_outputs': torch.sigmoid(task_outputs),
            'task_weights': task_weights
        }


class AIPLLabelSystem:
    """AIPL标签体系实现 - 将模型输出映射到业务标签"""
    STAGE_NAMES = ['Awareness', 'Interest', 'Purchase', 'Loyalty']
    STAGE_COLORS = {'Awareness': '认知期', 'Interest': '兴趣期', 'Purchase': '购买期', 'Loyalty': '忠诚期'}

    def __init__(self, model: STANLifecycleModel):
        self.model = model
        self.model.eval()

    def predict_stage(self, user_behaviors: List[UserBehavior]) -> Dict:
        behavior_tensor = self._preprocess_behaviors(user_behaviors)
        with torch.no_grad():
            outputs = self.model(behavior_tensor.unsqueeze(0))
        stage_idx = outputs['predicted_stage'].item()
        stage_probs = outputs['stage_probs'][0].numpy()
        task_names = ['点击概率', '加购概率', '购买概率']
        task_preds = outputs['task_outputs'][0].numpy()
        task_weights = outputs['task_weights'][0].numpy()
        return {
            'user_id': user_behaviors[0].user_id if user_behaviors else None,
            'stage': self.STAGE_NAMES[stage_idx],
            'stage_confidence': float(stage_probs[stage_idx]),
            'stage_distribution': {name: float(prob) for name, prob in zip(self.STAGE_NAMES, stage_probs)},
            'task_predictions': {name: float(pred) for name, pred in zip(task_names, task_preds)},
            'task_weights': {name: float(weight) for name, weight in zip(task_names, task_weights)},
            'recommendation_strategy': self._get_strategy(stage_idx)
        }

    def _preprocess_behaviors(self, behaviors: List[UserBehavior]) -> torch.Tensor:
        sorted_behaviors = sorted(behaviors, key=lambda x: x.timestamp)
        event_type_map = {'click': 0, 'cart': 1, 'purchase': 2, 'review': 3}
        baby_stage_map = {'0-6m': 0, '6-12m': 1, '1-3y': 2, '3y+': 3}
        tensor_data = []
        for b in sorted_behaviors[-50:]:
            tensor_data.append([b.category_id, event_type_map.get(b.event_type, 0), baby_stage_map.get(b.baby_stage, 0)])
        seq_len = 50
        if len(tensor_data) < seq_len:
            tensor_data = [[0, 0, 0]] * (seq_len - len(tensor_data)) + tensor_data
        return torch.tensor(tensor_data[-seq_len:], dtype=torch.long)

    def _get_strategy(self, stage_idx: int) -> str:
        strategies = [
            "策略：品牌教育+品类种草（推送育儿知识、新手妈妈指南）",
            "策略：优惠刺激+精准推荐（推送限时折扣、相似商品）",
            "策略：复购引导+会员权益（推送 replenish 提醒、积分活动）",
            "策略：忠诚度维护+口碑传播（推送新品预览、推荐有礼）"
        ]
        return strategies[stage_idx]


def create_sample_data() -> List[UserBehavior]:
    """创建示例用户行为数据"""
    np.random.seed(42)
    behaviors = []
    base_time = pd.Timestamp('2024-01-01')
    for i in range(30):
        timestamp = base_time + pd.Timedelta(hours=i * 2)
        if i < 10:
            event_type = 'click' if np.random.random() > 0.3 else 'cart'
        elif i < 20:
            event_type = np.random.choice(['click', 'cart', 'purchase'], p=[0.4, 0.4, 0.2])
        else:
            event_type = np.random.choice(['click', 'cart', 'purchase', 'review'], p=[0.3, 0.3, 0.3, 0.1])
        behaviors.append(UserBehavior(
            user_id='U12345', event_type=event_type, category_id=np.random.randint(100, 110),
            baby_stage='6-12m', timestamp=timestamp, value=np.random.uniform(50, 300) if event_type == 'purchase' else 0
        ))
    return behaviors


def test_lifecycle_model():
    """测试生命周期模型"""
    print("=" * 60)
    print("STAN 用户生命周期建模测试")
    print("=" * 60)
    num_categories = 200
    model = STANLifecycleModel(num_categories=num_categories, num_stages=4, embed_dim=64, hidden_dim=128, num_tasks=3)
    print("\n[OK] 模型创建成功")
    sample_behaviors = create_sample_data()
    print(f"\n[OK] 生成 {len(sample_behaviors)} 条示例行为数据")
    print("\n前5条行为:")
    for b in sample_behaviors[:5]:
        print(f"  - {b.timestamp.strftime('%m-%d %H:%M')}: {b.event_type} (品类{b.category_id}, {b.baby_stage})")
    aipl_system = AIPLLabelSystem(model)
    print("\n[OK] AIPL标签系统初始化成功")
    result = aipl_system.predict_stage(sample_behaviors)
    print("\n" + "=" * 60)
    print("预测结果")
    print("=" * 60)
    print(f"\n用户ID: {result['user_id']}")
    print(f"生命周期阶段: {result['stage']} ({aipl_system.STAGE_COLORS[result['stage']]})")
    print(f"阶段置信度: {result['stage_confidence']:.2%}")
    print(f"\n阶段分布:")
    for stage, prob in result['stage_distribution'].items():
        bar = "█" * int(prob * 20)
        print(f"  {stage:12s}: {prob:.1%} {bar}")
    print(f"\n任务预测 (点击率/加购率/购买率):")
    for task, pred in result['task_predictions'].items():
        print(f"  {task}: {pred:.2%}")
    print(f"\n任务权重 (阶段自适应):")
    for task, weight in result['task_weights'].items():
        print(f"  {task}: {weight:.2%}")
    print(f"\n推荐策略: {result['recommendation_strategy']}")
    print("\n" + "=" * 60)
    print("批量预测测试 (10个模拟用户)")
    print("=" * 60)
    stage_counts = defaultdict(int)
    for i in range(10):
        np.random.seed(42 + i)
        result = aipl_system.predict_stage(create_sample_data())
        stage_counts[result['stage']] += 1
        print(f"用户U{i:04d}: {result['stage']:12s} (置信度: {result['stage_confidence']:.1%})")
    print(f"\n用户分布:")
    for stage in aipl_system.STAGE_NAMES:
        count = stage_counts[stage]
        print(f"  {stage:12s}: {count}人 {'█' * count}")
    print("\n" + "=" * 60)
    print("测试完成 [OK]")
    print("=" * 60)
    return model, aipl_system


if __name__ == "__main__":
    model, aipl_system = test_lifecycle_model()
