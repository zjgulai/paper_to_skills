"""
STAN: Stage-Adaptive Network for User Lifecycle Modeling
用户生命周期自适应建模 - AIPL标签体系实现

主要组件:
- LifecycleStageEncoder: 生命周期阶段编码器
- TaskAdaptiveHead: 任务自适应头
- STANLifecycleModel: 完整STAN模型
- AIPLLabelSystem: AIPL标签体系实现

使用示例:
    from model import STANLifecycleModel, AIPLLabelSystem, UserBehavior

    model = STANLifecycleModel(num_categories=200)
    aipl = AIPLLabelSystem(model)
    result = aipl.predict_stage(user_behaviors)
"""

from .model import (
    UserBehavior,
    LifecycleStageEncoder,
    TaskAdaptiveHead,
    STANLifecycleModel,
    AIPLLabelSystem,
    create_sample_data,
    test_lifecycle_model
)

__all__ = [
    'UserBehavior',
    'LifecycleStageEncoder',
    'TaskAdaptiveHead',
    'STANLifecycleModel',
    'AIPLLabelSystem',
    'create_sample_data',
    'test_lifecycle_model'
]
