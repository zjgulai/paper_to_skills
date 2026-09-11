"""
Temporal Fusion Transformer for Demand Forecasting
母婴出海电商多水平时间序列预测

Reference: Lim et al. (2020)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Optional


class TFTDataset(Dataset):
    """TFT 数据集"""
    
    def __init__(self, static_features, time_varying_known, 
                 time_varying_observed, target, encoder_steps=30):
        self.static = static_features
        self.known = time_varying_known
        self.observed = time_varying_observed
        self.target = target
        self.encoder_steps = encoder_steps
        
    def __len__(self):
        return len(self.static)
    
    def __getitem__(self, idx):
        return {
            'static': torch.FloatTensor(self.static[idx]),
            'known': torch.FloatTensor(self.known[idx]),
            'observed': torch.FloatTensor(self.observed[idx]),
            'target': torch.FloatTensor(self.target[idx]),
            'encoder_steps': self.encoder_steps
        }


def generate_sample_data(n_samples=1000, encoder_steps=30, 
                         decoder_steps=7, n_categories=10):
    """生成母婴电商销量预测模拟数据"""
    np.random.seed(42)
    total_steps = encoder_steps + decoder_steps
    
    static = np.random.randn(n_samples, 3)
    static[:, 0] = np.random.randint(0, n_categories, n_samples)
    
    data_list = []
    for i in range(n_samples):
        t = np.arange(total_steps)
        trend = 0.1 * t + np.random.randn(total_steps) * 0.5
        seasonal = 5 * np.sin(2 * np.pi * t / 7)
        
        is_holiday = np.random.binomial(1, 0.1, total_steps)
        is_promo = np.zeros(total_steps)
        is_promo[encoder_steps:encoder_steps+3] = 1
        day_of_week = (t % 7) / 7.0
        
        known = np.column_stack([is_holiday, is_promo, day_of_week])
        
        base_demand = 20 + static[i, 0] * 5
        sales = base_demand + trend + seasonal + is_promo * 15 + np.random.randn(total_steps) * 3
        sales = np.maximum(sales, 0)
        page_views = sales * 10 + np.random.randn(total_steps) * 20
        
        observed = np.column_stack([sales[:encoder_steps], page_views[:encoder_steps]])
        observed = np.vstack([observed, np.zeros((decoder_steps, 2))])
        
        data_list.append({'static': static[i], 'known': known, 
                         'observed': observed, 'target': sales})
    
    static_arr = np.array([d['static'] for d in data_list])
    known_arr = np.array([d['known'] for d in data_list])
    observed_arr = np.array([d['observed'] for d in data_list])
    target_arr = np.array([d['target'] for d in data_list])
    
    split = int(0.8 * n_samples)
    train_data = {
        'static': static_arr[:split], 'known': known_arr[:split],
        'observed': observed_arr[:split], 'target': target_arr[:split]
    }
    test_data = {
        'static': static_arr[split:], 'known': known_arr[split:],
        'observed': observed_arr[split:], 'target': target_arr[split:]
    }
    return train_data, test_data


class GatedResidualNetwork(nn.Module):
    """门控残差网络 GRN"""
    
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.gate = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_size)
        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else None
    
    def forward(self, x, context=None):
        residual = x if self.skip is None else self.skip(x)
        hidden = self.fc1(x)
        if context is not None:
            hidden = hidden + context
        hidden = torch.relu(hidden)
        hidden = self.dropout(hidden)
        output = self.fc2(hidden)
        gate = torch.sigmoid(self.gate(hidden))
        return self.layer_norm(residual + gate * output)


class TemporalFusionTransformer(nn.Module):
    """TFT 完整模型"""
    
    def __init__(self, num_static=3, num_known=3, num_observed=2,
                 hidden_size=64, num_heads=4, num_layers=2, 
                 dropout=0.1, encoder_steps=30, decoder_steps=7):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder_steps = encoder_steps
        self.decoder_steps = decoder_steps
        self.quantiles = [0.1, 0.5, 0.9]
        
        self.static_encoder = nn.Linear(num_static, hidden_size)
        self.temporal_encoder = nn.LSTM(num_known + num_observed, 
                                        hidden_size, num_layers,
                                        batch_first=True, dropout=dropout)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, 
                                               dropout=dropout, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, 3)  # P10, P50, P90
    
    def forward(self, static, known, observed, encoder_steps):
        batch_size = static.size(0)
        static_emb = self.static_encoder(static).unsqueeze(1).expand(-1, known.size(1), -1)
        
        time_features = torch.cat([known, observed[:, :encoder_steps]], dim=-1)
        lstm_out, _ = self.temporal_encoder(time_features)
        
        attended, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        predictions = self.output_layer(attended)
        return predictions, {'attention_weights': attn_weights}
    
    def quantile_loss(self, predictions, targets):
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = targets - predictions[:, :, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(-1))
        return torch.mean(torch.sum(torch.cat(losses, dim=-1), dim=-1))


def train_tft(model, train_loader, val_loader, epochs=20, lr=1e-3, device='cpu'):
    """训练 TFT 模型"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            static = batch['static'].to(device)
            known = batch['known'].to(device)
            observed = batch['observed'].to(device)
            target = batch['target'].to(device)
            encoder_steps = batch['encoder_steps']
            
            optimizer.zero_grad()
            predictions, _ = model(static, known, observed, encoder_steps)
            decoder_pred = predictions[:, encoder_steps:, :]
            decoder_tgt = target[:, encoder_steps:]
            loss = model.quantile_loss(decoder_pred, decoder_tgt)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {np.mean(train_losses):.4f}")
    
    return model


def main():
    """主函数示例"""
    print("=" * 60)
    print("TFT 母婴出海电商销量预测")
    print("=" * 60)
    
    train_data, test_data = generate_sample_data(n_samples=500)
    print(f"\n训练样本: {len(train_data['static'])}")
    print(f"测试样本: {len(test_data['static'])}")
    
    train_dataset = TFTDataset(train_data['static'], train_data['known'],
                               train_data['observed'], train_data['target'])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = TemporalFusionTransformer()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    model = train_tft(model, train_loader, None, epochs=10, device=device)
    print("\n训练完成!")
    return model


if __name__ == "__main__":
    main()
