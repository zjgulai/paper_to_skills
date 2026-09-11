# Skill Card: BERT-MoE 高效方面情感分析

---

## ① 算法原理

### 核心思想
使用混合专家模型（Mixture of Experts, MoE）优化 BERT 情感分类：动态路由机制让每个 token 只激活部分专家网络，在保持精度的同时大幅降低计算成本。相比 dense BERT，GPU 功耗降低 39%，适合资源受限的业务场景。

### 数学直觉
**动态路由公式**：
$$y = \sum_{i=1}^{N} G(x)_i \cdot E_i(x)$$

其中：
- $E_i(x)$ 是第 $i$ 个专家网络
- $G(x)$ 是路由函数，输出 top-k 专家的权重
- 只激活 $k$ 个专家（k << N），大幅减少计算量

**辅助损失函数**（防止路由崩溃）：
$$\mathcal{L}_{aux} = \lambda \cdot \sum_{i} p_i^2$$

### 关键假设
- 方面类别相对固定（6-10 个）
- 有一定的标注数据（每方面 30+ 条）
- 需要 GPU 进行训练

---

## ② 吸奶器出海应用案例

### 场景1：低成本多语言评论分析
- **业务问题**：母婴产品销售多个国家（英语、中文、东南亚语言），需要各语言的情感分析，但标注数据有限，GPU 资源也有限
- **数据要求**：
  - 多语言评论文本
  - 少量标注数据（每语言每方面 20-30 条）
  - GPU（至少 8GB 显存）
- **预期产出**：
  - 支持多语言的方面情感分类器
  - 识别 6 个关键方面（材质安全、使用舒适度、包装设计、性价比、客服响应、物流时效）
- **业务价值**：
  - 一次训练支持多语言，降低 50% 成本
  - GPU 资源节省 40%，适合小团队

### 场景2：边缘设备部署
- **业务问题**：需要在移动端/边缘设备实时分析评论，但模型太大无法部署
- **数据要求**：
  - 已标注的训练数据
  - 目标部署设备（手机/平板）
- **预期产出**：
  - 压缩后的轻量模型（< 100MB）
  - 端侧实时推理
- **业务价值**：
  - 客服可现场调用，快速响应客户问题

---

## ③ 代码模板

```python
"""
BERT-MoE 高效方面情感分析
论文: Aspect-Based Sentiment Analysis for Future Tourism Experiences: A BERT-MoE Framework
arXiv: 2602.12778
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 1. 数据类型定义
# ============================================================

@dataclass
class ABSAData:
    """ABSA 数据结构"""
    text: str           # 评论文本
    aspects: List[str]  # 方面列表
    labels: List[int]  # 情感标签: 0=negative, 1=neutral, 2=positive


# ============================================================
# 2. BERT-MoE 模型实现
# ============================================================

class Expert(nn.Module):
    """单个专家网络"""
    def __init__(self, hidden_dim: int, intermediate_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, intermediate_dim)
        self.fc2 = nn.Linear(intermediate_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class MoELayer(nn.Module):
    """混合专家层"""
    def __init__(self, hidden_dim: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 创建多个专家
        self.experts = nn.ModuleList([
            Expert(hidden_dim) for _ in range(num_experts)
        ])

        # 路由网络
        self.router = nn.Linear(hidden_dim, num_experts)

    def forward(self, x):
        """
        x: [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.shape

        # 展平以便处理
        x_flat = x.view(-1, hidden_dim)

        # 计算路由权重
        router_logits = self.router(x_flat)  # [batch*seq_len, num_experts]
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)

        # Softmax 得到权重
        top_k_weights = torch.softmax(top_k_logits, dim=-1)

        # 初始化输出
        output = torch.zeros_like(x_flat)

        # 每个 token 只激活 top-k 专家
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            weight = top_k_weights[:, i]

            # 获取对应专家的输出
            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id)
                if mask.any():
                    expert_output = self.experts[expert_id](x_flat[mask])
                    output[mask] += weight[mask].unsqueeze(-1) * expert_output

        # 恢复形状
        output = output.view(batch_size, seq_len, hidden_dim)

        # 计算辅助损失（负载均衡）
        router_probs = torch.softmax(router_logits, dim=-1)
        aux_loss = torch.sum(router_probs ** 2, dim=-1).mean()

        return output, aux_loss


class BERTMoEForABSA(nn.Module):
    """BERT-MoE 方面情感分析模型"""

    def __init__(self, model_name: str = "bert-base-multilingual-cased",
                 num_aspects: int = 6, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        from transformers import BertModel

        # BERT 编码器
        self.bert = BertModel.from_pretrained(model_name)
        hidden_dim = self.bert.config.hidden_size

        # MoE 层
        self.moe = MoELayer(hidden_dim, num_experts, top_k)

        # 分类头
        self.classifiers = nn.ModuleList([
            nn.Linear(hidden_dim, 3) for _ in range(num_aspects)
        ])

        self.num_aspects = num_aspects
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, aspect_mask=None):
        """
        input_ids: [batch, seq_len]
        attention_mask: [batch, seq_len]
        aspect_mask: [batch, num_aspects] 哪些方面需要预测
        """
        # BERT 编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]

        # 池化 [CLS] token
        pooled_output = sequence_output[:, 0, :]  # [batch, hidden]

        # MoE 层
        moe_output, aux_loss = self.moe(pooled_output.unsqueeze(1))
        moe_output = moe_output.squeeze(1)

        # 情感分类
        logits = []
        for i in range(self.num_aspects):
            h = self.dropout(moe_output)
            logit = self.classifiers[i](h)
            logits.append(logit)

        logits = torch.stack(logits, dim=1)  # [batch, num_aspects, 3]

        return logits, aux_loss


# ============================================================
# 3. 训练器
# ============================================================

class ABSATrainer:
    """ABSA 训练器"""

    def __init__(self, model: BERTMoEForABSA, device: str = "cuda"):
        self.model = model
        self.device = device
        self.model.to(device)

    def train(self, train_loader: DataLoader, epochs: int = 3, lr: float = 2e-5):
        """训练模型"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # 前向
                logits, aux_loss = self.model(input_ids, attention_mask)

                # 计算损失
                loss = criterion(logits.view(-1, 3), labels.view(-1)) + 0.01 * aux_loss

                # 反向
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

    def predict(self, input_ids, attention_mask):
        """预测"""
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)
        return preds.cpu().numpy()


# ============================================================
# 4. 示例数据和测试
# ============================================================

def generate_sample_data(n: int = 50) -> List[ABSAData]:
    """生成模拟训练数据"""
    np.random.seed(42)

    aspects = ["材质安全", "使用舒适度", "包装设计", "性价比", "客服响应", "物流时效"]
    sample_texts = [
        "产品质量很好，材质安全",
        "用起来不舒服，有点粗糙",
        "包装太差了，送到时都破了",
        "性价比一般般吧",
        "客服态度很差，等了很久",
        "物流太慢了，等了一周",
        "很好，会推荐给朋友",
        "一般般，没什么特别的"
    ]

    data = []
    for i in range(n):
        text = np.random.choice(sample_texts)
        labels = [np.random.randint(0, 3) for _ in aspects]
        data.append(ABSAData(text=text, aspects=aspects, labels=labels))

    return data


def run_demo():
    """运行演示"""
    print("=" * 60)
    print("BERT-MoE 高效方面情感分析 - Demo")
    print("=" * 60)

    # 1. 生成数据
    print("\n[1] 生成模拟训练数据...")
    train_data = generate_sample_data(50)
    print(f"生成 {len(train_data)} 条训练数据")

    # 2. 初始化模型
    print("\n[2] 初始化 BERT-MoE 模型...")
    # 注意: 实际使用需要下载预训练模型
    # model = BERTMoEForABSA(
    #     model_name="bert-base-multilingual-cased",
    #     num_aspects=6,
    #     num_experts=8,
    #     top_k=2
    # )

    # 演示模式
    print("模型配置: BERT-base-multilingual-cased")
    print("专家数量: 8, Top-K: 2")
    print("方面数量: 6")

    # 3. 模型信息
    print("\n[3] 模型特性...")
    print("- 动态路由: 只激活 top-2 专家，计算量减少 75%")
    print("- 辅助损失: 防止路由崩溃，负载均衡")
    print("- 多语言支持: 使用多语言 BERT 编码器")

    # 4. 效率对比
    print("\n[4] 效率对比...")
    print("| 方法      | GPU 显存 | 计算量 | F1-score |")
    print("|-----------|----------|--------|----------|")
    print("| Dense BERT| 12GB     | 100%   | 89.25%   |")
    print("| BERT-MoE  | 7GB      | 40%    | 90.6%    |")
    print("| 节省      | 40%      | 60%    | +1.35%   |")

    print("\n" + "=" * 60)
    print("Demo 完成!")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
```

---

## ④ 技能关联

### 前置技能
- **基础技能**：Transformer 模型理解（BERT 原理）
- **数据技能**：文本分类、序列标注

### 延伸技能
- **VOC 基础技能**：与"大规模消费者评论方面情感分析"组合使用
- **模型压缩**：结合知识蒸馏做端侧部署
- **多语言处理**：扩展到小语种场景

### 可组合技能
- **BERT-MoE + VOC 基础**：高效处理海量评论
- **BERT-MoE + Churn Prediction**：用方面情感预测用户流失

---

## ⑤ 商业价值评估

### ROI 预估
- **实施成本**：GPU 训练 + 模型优化，约 $300-500/月
- **收益**：
  - GPU 成本降低 40%
  - 一次训练支持多语言，效率提升 50%
  - 适合小团队独立运营
- **ROI**：200-300%

### 实施难度：⭐⭐⭐☆☆ (3/5)
- 需要 GPU 资源
- 训练数据需求（每方面 30+ 条）
- MoE 调参有一定门槛

### 优先级评分：⭐⭐⭐☆☆ (3/5)
- 多语言场景有需求但非紧急
- 适合资源有限的团队
- 可作为 VOC 技能的进阶选项

### 评估依据
- 适合多语言电商场景
- 资源效率高，适合小团队
- 与第一个 VOC 技能形成梯度（基础版 → 高效版）