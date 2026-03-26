# 论文信息

## Paper 3: Time Series Forecasting with Transformer

### 论文信息
- **标题**: Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting
- **领域**: 时间序列 / 需求预测

### 核心算法
1. **LSTM/GRU**: 时序特征提取
2. **Attention Mechanism**: 注意力机制捕捉长期依赖
3. **Transformer**: 时序融合Transformer

### 关键公式
- Attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
- Multi-head: MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
- Temporal fusion: 结合历史编码和未来协变量

---

# 论文摘要

We introduce the Temporal Fusion Transformer (TFT), a novel attention-based architecture designed for multi-horizon forecasting with heterogeneous inputs. The key innovation is a multi-horizon forecasting model that combines gated residual networks, variable selection networks, and a self-attention mechanism to capture both temporal dependencies and static covariates. TFT achieves state-of-the-art performance on several benchmark datasets while maintaining interpretability through attention weight analysis.
