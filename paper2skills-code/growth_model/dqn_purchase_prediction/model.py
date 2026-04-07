"""
DQN-Inspired Deep Learning for Purchase Intent Prediction
DQN深度强化学习购买意图预测模型

论文: Predicting E-commerce Purchase Behavior using a DQN-Inspired Deep Learning Model
arXiv: 2506.17543 (2025)

核心创新:
- 结合DQN(Deep Q-Network)经验回放机制与LSTM时序建模
- Epsilon-Greedy探索策略处理类别不平衡
- 适应性强，适合高维稀疏电商数据
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Deque
from dataclasses import dataclass
from collections import deque, defaultdict
import random
import warnings
warnings.filterwarnings('ignore')


@dataclass
class UserSession:
    """用户会话数据"""
    user_id: str
    session_id: str
    features: np.ndarray  # 会话特征向量 (1,114维)
    sequence: List[int]   # 行为序列 (点击、浏览、加购等)
    has_purchase: bool
    timestamp: pd.Timestamp
    

class ExperienceReplayBuffer:
    """
    经验回放缓冲区 (DQN核心组件)
    存储和采样训练样本，打破数据相关性，提高样本效率
    """
    def __init__(self, capacity: int = 10000):
        self.buffer: Deque[Tuple] = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """存储经验样本"""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Tuple:
        """随机采样一批经验"""
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNLSTMNetwork(nn.Module):
    """
    DQN-inspired LSTM网络
    结合时序建模和强化学习的稳定性
    """
    def __init__(self, 
                 input_dim: int = 1114,  # 特征维度
                 seq_length: int = 20,    # 序列长度
                 hidden_dim: int = 256,
                 lstm_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        
        # LSTM编码器：建模用户行为序列
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # 注意力机制：关注关键时间步
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 全连接层：预测购买概率
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # 二分类：购买/不购买
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [batch_size, seq_length, input_dim]
        Returns:
            q_values: [batch_size, 2] (购买/不购买的Q值)
        """
        # LSTM编码
        lstm_out, _ = self.lstm(x)  # [B, T, H*2]
        
        # 注意力权重
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)  # [B, T, 1]
        
        # 加权求和
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [B, H*2]
        
        # 全连接预测
        q_values = self.fc(context)  # [B, 2]
        
        return q_values
    
    def get_purchase_probability(self, x: torch.Tensor) -> torch.Tensor:
        """获取购买概率"""
        with torch.no_grad():
            q_values = self.forward(x)
            probs = F.softmax(q_values, dim=-1)
            return probs[:, 1]  # 购买类的概率


class DQNPurchasePredictor:
    """
    DQN-inspired购买意图预测器
    整合经验回放、Epsilon-Greedy探索、动态学习
    """
    
    def __init__(self,
                 input_dim: int = 1114,
                 seq_length: int = 20,
                 hidden_dim: int = 256,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,  # 折扣因子
                 epsilon: float = 1.0,  # 探索率
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01,
                 buffer_capacity: int = 10000,
                 batch_size: int = 64):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Q网络和目标网络 (DQN双网络结构)
        self.q_network = DQNLSTMNetwork(input_dim, seq_length, hidden_dim).to(self.device)
        self.target_network = DQNLSTMNetwork(input_dim, seq_length, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ExperienceReplayBuffer(buffer_capacity)
        
        # DQN超参数
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        self.seq_length = seq_length
        self.input_dim = input_dim
        
    def preprocess_session(self, session: UserSession) -> np.ndarray:
        """预处理会话数据为模型输入"""
        # 特征标准化 (简化版，实际应用中应使用训练集的均值方差)
        features = session.features
        
        # 构建序列数据 (使用滑动窗口)
        seq_data = []
        for i in range(min(len(session.sequence), self.seq_length)):
            # 创建one-hot特征
            feat = np.zeros(self.input_dim)
            feat[:len(features)] = features
            feat[len(features) + session.sequence[i]] = 1  # 行为编码
            seq_data.append(feat)
        
        # 填充或截断
        while len(seq_data) < self.seq_length:
            seq_data.append(np.zeros(self.input_dim))
        
        return np.array(seq_data[-self.seq_length:])
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Epsilon-Greedy动作选择
        以epsilon概率随机探索，以1-epsilon概率选择最优动作
        """
        if training and random.random() < self.epsilon:
            # 探索：随机选择
            return random.randint(0, 1)
        else:
            # 利用：选择Q值最大的动作
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def train_step(self) -> Optional[float]:
        """执行一步训练"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # 从经验回放中采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # 当前Q值
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 目标Q值 (Double DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # 计算损失
        loss = F.mse_loss(current_q, target_q)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 衰减探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def predict(self, session: UserSession) -> Dict:
        """
        预测购买意图
        
        Returns:
            {
                'purchase_probability': float,
                'predicted_action': int,  # 0=不购买, 1=购买
                'q_values': List[float],
                'confidence': str  # 'high', 'medium', 'low'
            }
        """
        state = self.preprocess_session(session)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]
            probs = F.softmax(torch.tensor(q_values), dim=-1).numpy()
        
        purchase_prob = probs[1]
        predicted_action = 1 if purchase_prob > 0.5 else 0
        
        # 置信度
        if abs(purchase_prob - 0.5) > 0.3:
            confidence = 'high'
        elif abs(purchase_prob - 0.5) > 0.15:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return {
            'purchase_probability': float(purchase_prob),
            'predicted_action': predicted_action,
            'q_values': q_values.tolist(),
            'confidence': confidence
        }
    
    def train(self, sessions: List[UserSession], epochs: int = 10, update_target_every: int = 100):
        """训练模型"""
        print(f"训练数据: {len(sessions)} 个会话")
        
        # 构建经验回放缓冲区
        for i, session in enumerate(sessions[:-1]):
            state = self.preprocess_session(session)
            next_session = sessions[i + 1]
            next_state = self.preprocess_session(next_session)
            
            # 奖励设计
            reward = 1.0 if session.has_purchase else -0.1
            action = 1 if session.has_purchase else 0
            done = 1.0 if session.has_purchase else 0.0
            
            self.replay_buffer.push(state, action, reward, next_state, done)
        
        print(f"经验回放缓冲区: {len(self.replay_buffer)} 条记录")
        
        # 训练循环
        losses = []
        for epoch in range(epochs):
            epoch_losses = []
            for _ in range(len(self.replay_buffer) // self.batch_size):
                loss = self.train_step()
                if loss:
                    epoch_losses.append(loss)
            
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0
            losses.append(avg_loss)
            
            # 定期更新目标网络
            if epoch % update_target_every == 0:
                self.update_target_network()
            
            if epoch % 2 == 0:
                print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Epsilon={self.epsilon:.4f}")
        
        print("训练完成")
        return losses


class AIPLPurchaseScorer:
    """
    AIPL购买意向评分器
    将DQN预测结果映射到AIPL阶段的P阶段细分
    """
    
    def __init__(self, predictor: DQNPurchasePredictor):
        self.predictor = predictor
        
    def score_session(self, session: UserSession) -> Dict:
        """
        评分会话并输出AIPL标签
        
        Returns:
            {
                'user_id': str,
                'session_id': str,
                'aipr_stage': str,  # Awareness, Interest, Purchase-High, Purchase-Med, Purchase-Low
                'purchase_probability': float,
                'recommended_action': str,
                'urgency_score': int  # 1-10
            }
        """
        prediction = self.predictor.predict(session)
        prob = prediction['purchase_probability']
        
        # AIPL阶段细分
        if prob >= 0.8:
            stage = 'Purchase-High'
            action = '立即推送优惠券完成转化'
            urgency = 10
        elif prob >= 0.5:
            stage = 'Purchase-Med'
            action = '推送限时促销增加紧迫感'
            urgency = 7
        elif prob >= 0.2:
            stage = 'Purchase-Low'
            action = '教育内容+产品种草'
            urgency = 4
        else:
            stage = 'Interest'
            action = '品牌认知教育'
            urgency = 2
        
        return {
            'user_id': session.user_id,
            'session_id': session.session_id,
            'aipr_stage': stage,
            'purchase_probability': prob,
            'predicted_action': prediction['predicted_action'],
            'confidence': prediction['confidence'],
            'recommended_action': action,
            'urgency_score': urgency,
            'q_values': prediction['q_values']
        }


# ==================== 测试用例 ====================

def create_sample_sessions(n_samples: int = 100) -> List[UserSession]:
    """创建示例会话数据"""
    np.random.seed(42)
    sessions = []
    
    base_time = pd.Timestamp('2024-01-01')
    
    for i in range(n_samples):
        # 生成1,114维特征 (简化版)
        features = np.random.randn(100)  # 实际应用中应有1114维
        
        # 生成行为序列
        seq_length = np.random.randint(5, 30)
        sequence = np.random.randint(0, 10, seq_length).tolist()
        
        # 模拟购买标签 (购买率约30%)
        has_purchase = np.random.random() < 0.3
        
        sessions.append(UserSession(
            user_id=f'U{i//3:04d}',
            session_id=f'S{i:05d}',
            features=features,
            sequence=sequence,
            has_purchase=has_purchase,
            timestamp=base_time + pd.Timedelta(hours=i)
        ))
    
    return sessions


def test_dqn_purchase_prediction():
    """测试DQN购买预测模型"""
    print("=" * 70)
    print("DQN-Inspired 购买意图预测测试")
    print("=" * 70)
    
    # 1. 创建数据
    sessions = create_sample_sessions(200)
    train_sessions = sessions[:150]
    test_sessions = sessions[150:]
    
    purchase_count = sum(1 for s in sessions if s.has_purchase)
    print(f"\n[OK] 总样本: {len(sessions)} (购买: {purchase_count}, 未购买: {len(sessions)-purchase_count})")
    print(f"[OK] 训练集: {len(train_sessions)}, 测试集: {len(test_sessions)}")
    
    # 2. 创建模型
    predictor = DQNPurchasePredictor(
        input_dim=110,  # 100特征 + 10行为类别
        seq_length=20,
        hidden_dim=128,
        batch_size=32
    )
    print(f"[OK] 模型创建成功 (设备: {predictor.device})")
    
    # 3. 训练
    print("\n" + "=" * 70)
    print("模型训练")
    print("=" * 70)
    losses = predictor.train(train_sessions, epochs=5)
    
    # 4. 预测测试
    print("\n" + "=" * 70)
    print("购买意图预测 (测试集前10个)")
    print("=" * 70)
    
    scorer = AIPLPurchaseScorer(predictor)
    
    correct = 0
    for session in test_sessions[:10]:
        result = scorer.score_session(session)
        actual = "购买" if session.has_purchase else "未购买"
        predicted = "购买" if result['predicted_action'] == 1 else "未购买"
        match = "✓" if (session.has_purchase == (result['predicted_action'] == 1)) else "✗"
        
        if session.has_purchase == (result['predicted_action'] == 1):
            correct += 1
        
        print(f"\n用户{result['user_id']} 会话{result['session_id']}:")
        print(f"  AIPL阶段: {result['aipr_stage']}")
        print(f"  购买概率: {result['purchase_probability']:.2%}")
        print(f"  预测/实际: {predicted} / {actual} {match}")
        print(f"  置信度: {result['confidence']}")
        print(f"  建议动作: {result['recommended_action']}")
        print(f"  紧急度: {result['urgency_score']}/10")
    
    accuracy = correct / 10
    print(f"\n测试准确率: {accuracy:.1%}")
    
    # 5. 全量评估
    print("\n" + "=" * 70)
    print("全量测试集评估")
    print("=" * 70)
    
    all_correct = 0
    high_conf_correct = 0
    high_conf_total = 0
    
    for session in test_sessions:
        result = scorer.score_session(session)
        is_correct = (session.has_purchase == (result['predicted_action'] == 1))
        
        if is_correct:
            all_correct += 1
        
        if result['confidence'] == 'high':
            high_conf_total += 1
            if is_correct:
                high_conf_correct += 1
    
    total_accuracy = all_correct / len(test_sessions)
    high_conf_accuracy = high_conf_correct / high_conf_total if high_conf_total > 0 else 0
    
    print(f"总体准确率: {total_accuracy:.2%} ({all_correct}/{len(test_sessions)})")
    print(f"高置信度准确率: {high_conf_accuracy:.2%} ({high_conf_correct}/{high_conf_total})")
    print(f"高置信度占比: {high_conf_total/len(test_sessions):.1%}")
    
    print("\n" + "=" * 70)
    print("测试完成 [OK]")
    print("=" * 70)
    
    return predictor, scorer


if __name__ == "__main__":
    predictor, scorer = test_dqn_purchase_prediction()
