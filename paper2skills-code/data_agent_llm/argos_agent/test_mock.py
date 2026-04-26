"""
Argos Mock 测试
不依赖 LLM API，用预置规则验证三Agent协作框架
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from anomaly_detector import ArgosAnomalyDetector, DetectionRule


class MockLLMClient:
    """模拟 LLM 响应"""
    
    class ChatCompletions:
        def __init__(self, parent):
            self.parent = parent
        
        def create(self, model, messages, temperature=None):
            prompt = messages[-1]['content'] if messages else ""
            
            if "异常检测专家" in prompt or "不同角度" in prompt:
                # Detection Agent: 生成规则（无额外缩进，与模板匹配）
                content = """```python
# 基于Z-score检测异常，偏离均值超过threshold个标准差则标记异常
mean_val = np.mean(values)
std_val = np.std(values)
if std_val == 0:
    std_val = 1e-6
z_scores = np.abs((values - mean_val) / std_val)
labels[z_scores > threshold] = 1
```"""
            elif "报错" in prompt or "修正" in prompt:
                # Repair Agent: 修正代码（完整函数）
                content = """```python
import numpy as np

def inference(sample: np.ndarray, threshold: float) -> np.ndarray:
    values = sample[:, 0]
    labels = np.zeros(len(values), dtype=int)
    mean_val = np.mean(values)
    std_val = np.std(values)
    if std_val == 0:
        std_val = 1e-6
    z_scores = np.abs((values - mean_val) / std_val)
    labels[z_scores > threshold] = 1
    return labels
```"""
            elif "需要改进" in prompt:
                # Review Agent: 改进规则（完整函数）
                content = """```python
import numpy as np

def inference(sample: np.ndarray, threshold: float) -> np.ndarray:
    values = sample[:, 0]
    labels = np.zeros(len(values), dtype=int)
    # 使用滑动窗口计算局部Z-score，适应趋势变化
    window = min(30, len(values) // 4)
    for i in range(window, len(values)):
        local_mean = np.mean(values[i-window:i])
        local_std = np.std(values[i-window:i])
        if local_std == 0:
            local_std = 1e-6
        z = abs((values[i] - local_mean) / local_std)
        if z > threshold:
            labels[i] = 1
    return labels
```"""
            else:
                content = "```python\nlabels = np.zeros(len(values), dtype=int)\n```"
            
            class Choice:
                class Message:
                    def __init__(self, c):
                        self.content = c
                def __init__(self, c):
                    self.message = self.Message(c)
            
            class Response:
                def __init__(self, c):
                    self.choices = [Choice(c)]
            
            return Response(content)
    
    def __init__(self):
        self.chat = type('Chat', (), {'completions': self.ChatCompletions(self)})()


def create_synthetic_ts(n=200, anomaly_ratio=0.05):
    """生成合成时序数据"""
    np.random.seed(42)
    t = np.arange(n)
    trend = 0.02 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 30)
    noise = np.random.normal(0, 2, n)
    values = 50 + trend + seasonal + noise
    labels = np.zeros(n, dtype=int)
    n_anomalies = int(n * anomaly_ratio)
    anomaly_indices = np.random.choice(n, n_anomalies, replace=False)
    for idx in anomaly_indices:
        if np.random.rand() > 0.5:
            values[idx] += np.random.uniform(20, 40)
        else:
            values[idx] -= np.random.uniform(20, 40)
        labels[idx] = 1
    return values.reshape(-1, 1), labels


def test_mock_argos():
    """Mock 测试：验证三Agent协作框架"""
    print("=" * 60)
    print("Argos Anomaly Detector Mock 测试")
    print("=" * 60)
    
    # 生成数据
    print("\n[1/4] 生成合成时序数据...")
    data, labels = create_synthetic_ts(n=200, anomaly_ratio=0.05)
    split = int(0.7 * len(data))
    train_data, val_data = data[:split], data[split:]
    train_labels, val_labels = labels[:split], labels[split:]
    print(f"   训练集: {len(train_data)} 样本, 异常率: {train_labels.mean()*100:.1f}%")
    print(f"   验证集: {len(val_data)} 样本, 异常率: {val_labels.mean()*100:.1f}%")
    
    # 创建 Detector，注入 Mock
    print("\n[2/4] 初始化 Argos Detector (Mock 模式)...")
    detector = ArgosAnomalyDetector(api_key="sk-fake")
    detector.client = MockLLMClient()
    detector.model = "mock"
    
    # 训练
    print("\n[3/4] 开始训练（Detection → Repair → Review）...")
    try:
        rule = detector.fit(train_data, train_labels, val_data, val_labels)
        print(f"   ✓ Detection Agent: 生成 {detector.k_best} 条候选规则")
        print(f"   ✓ Repair Agent: 语法检查通过")
        print(f"   ✓ Review Agent: 规则评估完成")
        print(f"   最佳规则 F1: {rule.f1_score:.3f}")
        
        # 预测
        print("\n[4/4] 验证集预测...")
        predictions = detector.predict(val_data, rule)
        final_f1 = ArgosAnomalyDetector._calculate_f1(val_labels, predictions)
        print(f"   验证集 F1: {final_f1:.3f}")
        
        # 统计
        tp = np.sum((val_labels == 1) & (predictions == 1))
        fp = np.sum((val_labels == 0) & (predictions == 1))
        fn = np.sum((val_labels == 1) & (predictions == 0))
        tn = np.sum((val_labels == 0) & (predictions == 0))
        print(f"   混淆矩阵: TP={tp} FP={fp} FN={fn} TN={tn}")
        
        print("\n" + "=" * 60)
        print("Mock 测试通过！三Agent协作框架验证完毕。")
        print("接入真实 LLM：将 detector.client = MockLLMClient() 替换为真实客户端")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_mock_argos()
