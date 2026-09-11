"""
Deep Learning for Customer Churn Prediction
基于深度学习的客户流失预测

Reference: Spanoudes et al. (2017)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')


def generate_maternity_ecommerce_data(n_samples=10000):
    """生成母婴电商用户行为数据"""
    np.random.seed(42)
    
    data = {
        'user_id': range(n_samples),
        # RFM 特征
        'recency_days': np.random.exponential(30, n_samples),  # 距上次购买天数
        'frequency': np.random.poisson(5, n_samples),  # 购买频次
        'monetary': np.random.lognormal(5, 1, n_samples),  # 消费金额
        
        # 行为特征
        'days_since_registration': np.random.randint(1, 730, n_samples),
        'avg_order_value': np.random.lognormal(4, 0.8, n_samples),
        'total_orders': np.random.poisson(8, n_samples),
        'web_visits_30d': np.random.poisson(10, n_samples),
        'app_opens_30d': np.random.poisson(5, n_samples),
        
        # 产品偏好特征
        'pct_diaper_orders': np.random.beta(2, 3, n_samples),
        'pct_formula_orders': np.random.beta(1, 4, n_samples),
        'pct_clothing_orders': np.random.beta(3, 2, n_samples),
        
        # 渠道特征
        'is_mobile_user': np.random.binomial(1, 0.6, n_samples),
        'is_app_user': np.random.binomial(1, 0.4, n_samples),
        
        # 客服交互
        'complaints_90d': np.random.poisson(0.5, n_samples),
        'returns_90d': np.random.poisson(0.3, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # 生成流失标签（基于业务逻辑）
    churn_score = (
        (df['recency_days'] > 60).astype(int) * 0.4 +
        (df['frequency'] < 2).astype(int) * 0.3 +
        (df['web_visits_30d'] == 0).astype(int) * 0.2 +
        (df['complaints_90d'] > 2).astype(int) * 0.1
    )
    df['churned'] = (churn_score > 0.5).astype(int)
    
    # 添加噪声
    flip_mask = np.random.random(n_samples) < 0.05
    df.loc[flip_mask, 'churned'] = 1 - df.loc[flip_mask, 'churned']
    
    return df


class ChurnDataset(Dataset):
    """流失预测数据集"""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class DeepChurnPredictor(nn.Module):
    """深度流失预测模型"""
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train_churn_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
    """训练流失预测模型"""
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    
    best_val_auc = 0
    history = {'train_loss': [], 'val_auc': []}
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # 验证
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                outputs = model(batch_features)
                val_preds.extend(outputs.cpu().numpy())
                val_labels.extend(batch_labels.numpy())
        
        val_auc = roc_auc_score(val_labels, val_preds)
        history['train_loss'].append(np.mean(train_losses))
        history['val_auc'].append(val_auc)
        
        scheduler.step(val_auc)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), 'best_churn_model.pt')
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {np.mean(train_losses):.4f}, Val AUC: {val_auc:.4f}")
    
    return history


def predict_churn_risk(model, features, device='cpu'):
    """预测流失风险"""
    model.eval()
    with torch.no_grad():
        features_tensor = torch.FloatTensor(features).to(device)
        predictions = model(features_tensor).cpu().numpy()
    return predictions.flatten()


def get_high_risk_users(df, churn_probs, top_k=100):
    """获取高风险用户列表"""
    df = df.copy()
    df['churn_probability'] = churn_probs
    df['risk_segment'] = pd.cut(df['churn_probability'], 
                                 bins=[0, 0.3, 0.6, 0.8, 1.0],
                                 labels=['Low', 'Medium', 'High', 'Critical'])
    
    high_risk = df.nlargest(top_k, 'churn_probability')
    return high_risk[['user_id', 'recency_days', 'frequency', 'monetary', 
                      'churn_probability', 'risk_segment']]


def main():
    """主函数：母婴电商流失预测示例"""
    print("=" * 60)
    print("Deep Learning Customer Churn Prediction")
    print("母婴出海电商用户流失预测")
    print("=" * 60)
    
    # 1. 生成数据
    print("\n[1] 生成模拟数据...")
    df = generate_maternity_ecommerce_data(n_samples=10000)
    print(f"总用户数: {len(df)}")
    print(f"流失用户: {df['churned'].sum()} ({df['churned'].mean()*100:.1f}%)")
    
    # 2. 特征工程
    feature_cols = [col for col in df.columns if col not in ['user_id', 'churned']]
    X = df[feature_cols].values
    y = df['churned'].values
    
    # 3. 处理类别不平衡
    print("\n[2] 处理类别不平衡...")
    undersampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    X_resampled, y_resampled = undersampler.fit_resample(X, y)
    print(f"欠采样后 - 总样本: {len(X_resampled)}, 流失比例: {y_resampled.mean()*100:.1f}%")
    
    # 4. 划分数据集
    X_train, X_val, y_train, y_val = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )
    
    # 5. 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 6. 创建数据加载器
    train_dataset = ChurnDataset(X_train_scaled, y_train)
    val_dataset = ChurnDataset(X_val_scaled, y_val)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256)
    
    # 7. 初始化模型
    print("\n[3] 训练深度流失预测模型...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    model = DeepChurnPredictor(input_dim=len(feature_cols), hidden_dims=[128, 64, 32])
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    history = train_churn_model(model, train_loader, val_loader, epochs=50, device=device)
    
    # 8. 评估
    print("\n[4] 模型评估...")
    model.load_state_dict(torch.load('best_churn_model.pt'))
    
    val_probs = predict_churn_risk(model, X_val_scaled, device)
    val_preds = (val_probs > 0.5).astype(int)
    
    print("\n分类报告:")
    print(classification_report(y_val, val_preds, target_names=['Retained', 'Churned']))
    print(f"ROC-AUC Score: {roc_auc_score(y_val, val_probs):.4f}")
    
    # 9. 高风险用户识别
    print("\n[5] 高风险用户示例 (Top 10):")
    df_val = pd.DataFrame(X_val, columns=feature_cols)
    df_val['churned'] = y_val
    high_risk = get_high_risk_users(df_val, val_probs, top_k=10)
    print(high_risk.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("流失预测完成!")
    print("=" * 60)
    
    return model, scaler


if __name__ == "__main__":
    model, scaler = main()
