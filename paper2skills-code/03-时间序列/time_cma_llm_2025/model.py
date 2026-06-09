"""
TimeCMA: Towards LLM-Empowered Multivariate Time Series Forecasting
        via Cross-Modality Alignment
母婴出海电商 - 融合 LLM 世界知识的多变量时序预测

Reference: arXiv 2406.01638 (v5, 2025-03)

架构亮点:
1. 双分支编码: 数值分支(轻量 Patch Encoder) + LLM Prompt 分支
2. 跨模态对齐: 对比学习将两路 Embedding 对齐
3. 尾部 Token 压缩: LLM 输出只取最后一个 Token，极省显存
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────
# 1. 轻量数值分支：Patch-based 时序编码器
# ─────────────────────────────────────────────

class PatchEncoder(nn.Module):
    """将时序切成等长 Patch，再过 Transformer 编码，提取"纯净但较弱"的周期特征。"""

    def __init__(
        self,
        seq_len: int = 96,
        patch_len: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_len = patch_len
        n_patches = seq_len // patch_len  # 完整 Patch 数量

        # Patch 线性投影
        self.patch_proj = nn.Linear(patch_len, d_model)
        # 位置编码
        self.pos_emb = nn.Parameter(torch.randn(1, n_patches, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, n_vars) 多变量时序输入
        Returns:
            emb: (B, d_model) 全局时序表征（均值池化）
        """
        B, T, C = x.shape
        # 对每个变量分别切 Patch，然后拼接
        patches_list = []
        for c in range(C):
            var_series = x[:, :, c]  # (B, T)
            # 切 Patch: (B, n_patches, patch_len)
            n_p = T // self.patch_len
            var_patches = var_series[:, : n_p * self.patch_len].reshape(B, n_p, self.patch_len)
            patches_list.append(var_patches)

        # 多变量拼接后在 token 维度上共享: (B, n_patches * C, patch_len)
        patches = torch.cat(patches_list, dim=1)  # (B, n_patches*C, patch_len)

        # 线性投影到 d_model
        h = self.patch_proj(patches)  # (B, n_patches*C, d_model)

        # 简单位置编码：重复以适配 n_patches*C 长度
        n_tokens = h.size(1)
        base_pe = self.pos_emb.repeat(1, C, 1)  # (1, n_patches*C, d_model)
        if base_pe.size(1) != n_tokens:
            base_pe = base_pe[:, :n_tokens, :]
        h = h + base_pe

        h = self.transformer(h)  # (B, n_patches*C, d_model)
        h = self.norm(h)
        emb = h.mean(dim=1)  # (B, d_model) 均值池化
        return emb


# ─────────────────────────────────────────────
# 2. LLM Prompt 分支（Mock）
#    生产中接 LLaMA/GPT-2 等冻结 LLM；
#    这里用可学习矩阵模拟"尾部 Token 压缩"黑科技。
# ─────────────────────────────────────────────

class MockLLMBranch(nn.Module):
    """
    模拟冻结 LLM 的 Prompt 分支。
    - 接受数值序列 + 自然语言 Prompt 向量
    - 输出"尾部 Token"对应的语义表征（(B, d_llm)）
    生产替换: 用 transformers.AutoModel 加载真实 LLM，
              取最后一层最后一个 Token 的隐层输出。
    """

    def __init__(self, d_model: int = 128, d_llm: int = 256, prompt_dim: int = 64):
        super().__init__()
        # 数值序列映射到 LLM 输入空间
        self.num_proj = nn.Linear(d_model, d_llm)
        # Prompt 向量（代替真实 tokenize + embedding）
        self.prompt_proj = nn.Linear(prompt_dim, d_llm)
        # 模拟 LLM 前向（2 层 MLP 近似 FFN 语义变换）
        self.ffn = nn.Sequential(
            nn.Linear(d_llm, d_llm * 2),
            nn.GELU(),
            nn.Linear(d_llm * 2, d_llm),
        )
        self.norm = nn.LayerNorm(d_llm)

    def forward(self, num_emb: torch.Tensor, prompt_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            num_emb:    (B, d_model) 来自 PatchEncoder
            prompt_vec: (B, prompt_dim) 业务 Prompt 的稠密向量表示
        Returns:
            last_token: (B, d_llm) "尾部 Token 压缩"后的语义表征
        """
        h = self.num_proj(num_emb) + self.prompt_proj(prompt_vec)
        h = self.ffn(h)
        last_token = self.norm(h)  # (B, d_llm) 对应 LLM 尾部 Token
        return last_token


# ─────────────────────────────────────────────
# 3. 跨模态对齐模块（对比学习）
# ─────────────────────────────────────────────

class CrossModalityAlignment(nn.Module):
    """
    将数值分支 Embedding 和 LLM 分支 Embedding 映射到统一对齐空间。
    使用 InfoNCE 风格对比损失：
      L = -log( exp(cos(z_n, z_l) / τ) / Σ_j exp(cos(z_n, z_l_j) / τ) )
    让同一样本的数值表征和 LLM 表征更接近，不同样本的相互排斥。
    """

    def __init__(self, d_model: int = 128, d_llm: int = 256, d_align: int = 128, temperature: float = 0.07):
        super().__init__()
        self.num_proj = nn.Linear(d_model, d_align)
        self.llm_proj = nn.Linear(d_llm, d_align)
        self.temperature = temperature

    def forward(
        self, num_emb: torch.Tensor, llm_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            num_emb: (B, d_model)
            llm_emb: (B, d_llm)
        Returns:
            fused:       (B, d_align)  融合后的表征（逐元素加权均值）
            align_loss:  scalar       InfoNCE 对比损失
        """
        z_n = F.normalize(self.num_proj(num_emb), dim=-1)   # (B, d_align)
        z_l = F.normalize(self.llm_proj(llm_emb), dim=-1)   # (B, d_align)

        # InfoNCE 损失
        B = z_n.size(0)
        sim_matrix = torch.matmul(z_n, z_l.T) / self.temperature  # (B, B)
        labels = torch.arange(B, device=z_n.device)
        loss_n2l = F.cross_entropy(sim_matrix, labels)
        loss_l2n = F.cross_entropy(sim_matrix.T, labels)
        align_loss = (loss_n2l + loss_l2n) / 2.0

        # 融合：加权均值（LLM 特征占 0.6，数值特征占 0.4，可调）
        fused = 0.4 * z_n + 0.6 * z_l  # (B, d_align)
        return fused, align_loss


# ─────────────────────────────────────────────
# 4. 预测头
# ─────────────────────────────────────────────

class ForecastHead(nn.Module):
    """将对齐后的表征映射到预测步的输出序列。"""

    def __init__(self, d_align: int = 128, pred_len: int = 7, n_vars: int = 1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_align, d_align * 2),
            nn.ReLU(),
            nn.Linear(d_align * 2, pred_len * n_vars),
        )
        self.pred_len = pred_len
        self.n_vars = n_vars

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused: (B, d_align)
        Returns:
            pred: (B, pred_len, n_vars)
        """
        out = self.head(fused)  # (B, pred_len * n_vars)
        pred = out.reshape(-1, self.pred_len, self.n_vars)
        return pred


# ─────────────────────────────────────────────
# 5. TimeCMA 整体模型
# ─────────────────────────────────────────────

class TimeCMA(nn.Module):
    """
    TimeCMA 完整模型：
      PatchEncoder → MockLLMBranch → CrossModalityAlignment → ForecastHead
    """

    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 7,
        n_vars: int = 3,
        patch_len: int = 16,
        d_model: int = 128,
        d_llm: int = 256,
        d_align: int = 128,
        prompt_dim: int = 64,
        temperature: float = 0.07,
        alpha: float = 0.1,  # 对比损失权重
    ):
        super().__init__()
        self.alpha = alpha
        # 调整 seq_len 为 patch_len 的整数倍
        self.seq_len = (seq_len // patch_len) * patch_len

        self.num_encoder = PatchEncoder(
            seq_len=self.seq_len,
            patch_len=patch_len,
            d_model=d_model,
            n_heads=4,
            n_layers=2,
        )
        self.llm_branch = MockLLMBranch(
            d_model=d_model,
            d_llm=d_llm,
            prompt_dim=prompt_dim,
        )
        self.alignment = CrossModalityAlignment(
            d_model=d_model,
            d_llm=d_llm,
            d_align=d_align,
            temperature=temperature,
        )
        self.forecast_head = ForecastHead(
            d_align=d_align,
            pred_len=pred_len,
            n_vars=n_vars,
        )

    def forward(
        self,
        x_num: torch.Tensor,
        prompt_vec: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_num:      (B, seq_len, n_vars) 归一化后的数值时序
            prompt_vec: (B, prompt_dim)      业务事件 Prompt 稠密向量
        Returns:
            dict 含:
              'pred'       : (B, pred_len, n_vars) 预测值
              'align_loss' : scalar                 跨模态对齐损失
        """
        # 确保输入序列长度对齐
        x_num = x_num[:, : self.seq_len, :]

        # ① 数值分支
        num_emb = self.num_encoder(x_num)            # (B, d_model)

        # ② LLM Prompt 分支（尾部 Token 压缩）
        llm_emb = self.llm_branch(num_emb, prompt_vec)  # (B, d_llm)

        # ③ 跨模态对齐
        fused, align_loss = self.alignment(num_emb, llm_emb)  # (B, d_align)

        # ④ 预测
        pred = self.forecast_head(fused)              # (B, pred_len, n_vars)

        return {"pred": pred, "align_loss": align_loss}


# ─────────────────────────────────────────────
# 6. 业务 Prompt 编码器（文本 → 稠密向量）
# ─────────────────────────────────────────────

def encode_prompt(prompt_texts: List[str], prompt_dim: int = 64) -> torch.Tensor:
    """
    将业务事件文本转换为固定维度向量。
    生产环境: 接 sentence-transformers 或 text-embedding-ada-002。
    这里用确定性哈希 mock，保证相同文本输出相同向量。
    """
    vectors = []
    for text in prompt_texts:
        # 基于文本内容生成确定性伪随机向量
        seed = abs(hash(text)) % (2**31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(prompt_dim).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        vectors.append(vec)
    return torch.tensor(np.array(vectors), dtype=torch.float32)


# ─────────────────────────────────────────────
# 7. 生成模拟数据（母婴出海跨境电商场景）
# ─────────────────────────────────────────────

def generate_ecommerce_data(
    n_samples: int = 200,
    seq_len: int = 96,
    pred_len: int = 7,
    n_vars: int = 3,
    prompt_dim: int = 64,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """
    生成母婴出海电商多变量时序模拟数据。
    变量: [日销量, 页面浏览量, 购物车添加数]
    包含: 周期性趋势 + 促销脉冲 + 外部事件（如超级碗、政策变化）
    """
    np.random.seed(seed)
    T = seq_len + pred_len

    # 定义业务事件 Prompt（跨境电商真实场景）
    event_prompts = [
        "正常销售期，无特殊事件",
        "超级碗期间，零食/饮料类销量预计上涨30%",
        "加州限制塑料法案1023本周生效，影响塑料包装商品销量",
        "亚马逊Prime会员日促销开始，全品类流量显著提升",
        "母亲节临近，母婴用品需求上升，预计提升25%",
    ]

    x_num_list, prompt_vec_list, y_list = [], [], []
    for i in range(n_samples):
        t = np.arange(T)
        # 多变量时序生成
        trend = 0.05 * t
        weekly = 8.0 * np.sin(2 * np.pi * t / 7)
        monthly = 3.0 * np.sin(2 * np.pi * t / 30)
        noise = np.random.randn(T) * 2.0

        # 选择事件
        event_idx = np.random.randint(0, len(event_prompts))
        event_prompt = event_prompts[event_idx]
        event_boost = 0.0
        if "超级碗" in event_prompt:
            event_boost = 15.0
        elif "法案" in event_prompt:
            event_boost = -20.0
        elif "Prime" in event_prompt:
            event_boost = 25.0
        elif "母亲节" in event_prompt:
            event_boost = 18.0

        # 事件影响集中在预测期
        event_signal = np.zeros(T)
        event_signal[-pred_len:] = event_boost

        sales = 50.0 + trend + weekly + monthly + noise + event_signal
        sales = np.maximum(sales, 0)

        page_views = sales * 8.0 + np.random.randn(T) * 10
        cart_adds = sales * 0.3 + np.random.randn(T) * 2

        x_full = np.stack([sales, page_views, cart_adds], axis=-1)  # (T, 3)

        # 归一化
        mean_val = x_full[:seq_len].mean(axis=0, keepdims=True) + 1e-8
        std_val = x_full[:seq_len].std(axis=0, keepdims=True) + 1e-8
        x_norm = (x_full - mean_val) / std_val

        x_num_list.append(x_norm[:seq_len])          # (seq_len, n_vars)
        y_list.append(x_norm[seq_len:, :])            # (pred_len, n_vars)
        prompt_vec_list.append(event_prompt)

    x_num = torch.tensor(np.array(x_num_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.float32)
    prompt_vecs = encode_prompt(prompt_vec_list, prompt_dim=prompt_dim)

    # 划分训练/测试集
    split = int(0.8 * n_samples)
    return {
        "train_x": x_num[:split],
        "train_prompt": prompt_vecs[:split],
        "train_y": y[:split],
        "test_x": x_num[split:],
        "test_prompt": prompt_vecs[split:],
        "test_y": y[split:],
    }


# ─────────────────────────────────────────────
# 8. 训练函数
# ─────────────────────────────────────────────

def train_timecma(
    model: TimeCMA,
    data: Dict[str, torch.Tensor],
    n_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> List[float]:
    """
    简单训练循环，返回每轮 loss。
    损失 = MSE(预测, 真实) + alpha * 对比对齐损失
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x_train = data["train_x"]
    p_train = data["train_prompt"]
    y_train = data["train_y"]

    n = x_train.size(0)
    losses = []

    model.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start: start + batch_size]
            xb = x_train[idx]
            pb = p_train[idx]
            yb = y_train[idx]

            optimizer.zero_grad()
            out = model(xb, pb)
            pred = out["pred"]           # (B, pred_len, n_vars)
            align_loss = out["align_loss"]

            mse_loss = F.mse_loss(pred, yb)
            total_loss = mse_loss + model.alpha * align_loss

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  Epoch [{epoch + 1:02d}/{n_epochs}] loss={avg_loss:.4f}")

    return losses


# ─────────────────────────────────────────────
# 9. 评估函数
# ─────────────────────────────────────────────

def evaluate_timecma(
    model: TimeCMA,
    data: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """计算测试集 MAE / MSE / MAPE。"""
    model.eval()
    with torch.no_grad():
        out = model(data["test_x"], data["test_prompt"])
        pred = out["pred"].numpy()       # (N, pred_len, n_vars)
        true = data["test_y"].numpy()   # (N, pred_len, n_vars)

    mae = float(np.mean(np.abs(pred - true)))
    mse = float(np.mean((pred - true) ** 2))
    # MAPE：避免除零
    mask = np.abs(true) > 1e-6
    mape = float(np.mean(np.abs((pred[mask] - true[mask]) / true[mask]))) * 100

    return {"MAE": round(mae, 4), "MSE": round(mse, 4), "MAPE_pct": round(mape, 2)}


# ─────────────────────────────────────────────
# 10. 业务接口：单次预测（含 Prompt 注入）
# ─────────────────────────────────────────────

def predict_with_event(
    model: TimeCMA,
    history_data: np.ndarray,
    event_description: str,
    prompt_dim: int = 64,
) -> np.ndarray:
    """
    业务层调用接口：给定历史序列 + 事件描述，返回预测结果。

    Args:
        model:             已训练的 TimeCMA 实例
        history_data:      (seq_len, n_vars) numpy 历史数据（已归一化）
        event_description: 业务事件文本，如"母亲节临近，促销力度+20%"
        prompt_dim:        Prompt 向量维度

    Returns:
        pred_values: (pred_len, n_vars) numpy 预测值
    """
    model.eval()
    x = torch.tensor(history_data[np.newaxis, :, :], dtype=torch.float32)
    p = encode_prompt([event_description], prompt_dim=prompt_dim)
    with torch.no_grad():
        out = model(x, p)
    return out["pred"].squeeze(0).numpy()


# ─────────────────────────────────────────────
# 11. 自测（pytest 兼容 + 直接运行均可）
# ─────────────────────────────────────────────

def _test_patch_encoder():
    """测试 PatchEncoder 输出形状。"""
    enc = PatchEncoder(seq_len=96, patch_len=16, d_model=64)
    x = torch.randn(4, 96, 3)
    out = enc(x)
    assert out.shape == (4, 64), f"PatchEncoder 输出形状错误: {out.shape}"
    print("  [PASS] PatchEncoder: (4, 96, 3) -> (4, 64)")


def _test_llm_branch():
    """测试 MockLLMBranch 输出形状。"""
    branch = MockLLMBranch(d_model=64, d_llm=128, prompt_dim=32)
    num_emb = torch.randn(4, 64)
    prompt_vec = torch.randn(4, 32)
    out = branch(num_emb, prompt_vec)
    assert out.shape == (4, 128), f"LLMBranch 输出形状错误: {out.shape}"
    print("  [PASS] MockLLMBranch: num(4,64) + prompt(4,32) -> (4,128)")


def _test_cross_modality_alignment():
    """测试 CrossModalityAlignment 损失值合法 & 融合形状。"""
    align = CrossModalityAlignment(d_model=64, d_llm=128, d_align=64, temperature=0.07)
    num_emb = torch.randn(8, 64)
    llm_emb = torch.randn(8, 128)
    fused, loss = align(num_emb, llm_emb)
    assert fused.shape == (8, 64), f"Alignment fused 形状错误: {fused.shape}"
    assert loss.item() > 0, "align_loss 应为正数"
    print(f"  [PASS] CrossModalityAlignment: fused(8,64), align_loss={loss.item():.4f}")


def _test_timecma_forward():
    """测试 TimeCMA 端到端前向传播。"""
    model = TimeCMA(
        seq_len=96, pred_len=7, n_vars=3,
        patch_len=16, d_model=64, d_llm=128, d_align=64, prompt_dim=32,
    )
    x = torch.randn(4, 96, 3)
    p = torch.randn(4, 32)
    out = model(x, p)
    assert out["pred"].shape == (4, 7, 3), f"预测形状错误: {out['pred'].shape}"
    assert out["align_loss"].item() > 0
    print(f"  [PASS] TimeCMA forward: pred(4,7,3), align_loss={out['align_loss'].item():.4f}")


def _test_encode_prompt():
    """测试 Prompt 编码确定性。"""
    texts = ["超级碗期间促销", "正常销售期"]
    v1 = encode_prompt(texts, prompt_dim=64)
    v2 = encode_prompt(texts, prompt_dim=64)
    assert torch.allclose(v1, v2), "Prompt 编码应具有确定性"
    assert v1.shape == (2, 64), f"Prompt 向量形状错误: {v1.shape}"
    print("  [PASS] encode_prompt: 确定性 + 形状 (2, 64)")


def _test_training_and_evaluation():
    """测试训练循环和评估指标是否正常工作。"""
    model = TimeCMA(
        seq_len=96, pred_len=7, n_vars=3,
        patch_len=16, d_model=64, d_llm=128, d_align=64,
        prompt_dim=64, alpha=0.1,
    )
    data = generate_ecommerce_data(
        n_samples=60, seq_len=96, pred_len=7, n_vars=3, prompt_dim=64, seed=0,
    )
    print("  开始训练 (5 轮)...")
    losses = train_timecma(model, data, n_epochs=5, batch_size=16, lr=1e-3)
    assert len(losses) == 5, "应返回5轮损失"
    assert losses[-1] < losses[0] * 5, "损失应维持在合理范围内"

    metrics = evaluate_timecma(model, data)
    print(f"  评估指标: {metrics}")
    assert "MAE" in metrics and "MSE" in metrics and "MAPE_pct" in metrics
    assert metrics["MAE"] < 100, f"MAE 过大: {metrics['MAE']}"
    print("  [PASS] 训练 & 评估正常")


def _test_predict_with_event():
    """测试业务接口 predict_with_event。"""
    model = TimeCMA(
        seq_len=96, pred_len=7, n_vars=3,
        patch_len=16, d_model=64, d_llm=128, d_align=64, prompt_dim=64,
    )
    history = np.random.randn(96, 3).astype(np.float32)
    pred = predict_with_event(
        model, history,
        event_description="亚马逊 Prime 日促销，母婴品类流量提升 30%",
        prompt_dim=64,
    )
    assert pred.shape == (7, 3), f"predict_with_event 输出形状错误: {pred.shape}"
    print(f"  [PASS] predict_with_event: (7, 3), 前3行预测值=\n    {pred[:3]}")


def run_all_tests():
    """运行全部自测用例。"""
    print("=" * 60)
    print("TimeCMA 自测开始")
    print("=" * 60)

    tests = [
        ("PatchEncoder", _test_patch_encoder),
        ("MockLLMBranch", _test_llm_branch),
        ("CrossModalityAlignment", _test_cross_modality_alignment),
        ("TimeCMA Forward", _test_timecma_forward),
        ("encode_prompt", _test_encode_prompt),
        ("Training & Evaluation", _test_training_and_evaluation),
        ("predict_with_event", _test_predict_with_event),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\n[{name}]")
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] AssertionError: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"结果: {passed} 通过 / {failed} 失败 / {len(tests)} 总计")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
