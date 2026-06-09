"""
BERT-SRL — 基于 BERT 的语义角色标注
基于论文: Shi & Lin "Simple BERT Models for Relation Extraction and Semantic Role Labeling" (2019)

核心能力:
1. 谓词检测 — 识别句子中的谓词/动作
2. 谓词消歧 — 确定谓词的具体语义
3. 论元识别 — 检测谓词的论元跨度
4. 论元分类 — 为论元分配语义角色 (ARG0, ARG1, ...)

母婴电商场景: 从评论中抽取 "谁-做了什么-针对什么-结果如何"
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SRLFrame:
    """语义角色标注框架"""
    predicate: str          # 谓词 (动作/状态)
    predicate_span: Tuple[int, int]
    arguments: List[Dict]   # [{role, text, span}, ...]
    sentence: str


class BERTSRLModel(nn.Module):
    """
    简化的 BERT-SRL 模型

    论文核心: 使用 [CLS] sentence [SEP] predicate [SEP] 的输入格式，
    让 BERT 编码器在 predicate-aware 的条件下预测论元标签。
    """

    def __init__(self, vocab_size: int = 30522, hidden_dim: int = 768, num_roles: int = 10):
        super().__init__()
        # 简化的 embedding 层（生产环境使用 transformers.BertModel）
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(512, hidden_dim)

        # BiLSTM 编码器
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)

        # 谓词指示器嵌入
        self.predicate_indicator = nn.Embedding(2, hidden_dim)

        # 论元分类器
        self.classifier = nn.Linear(hidden_dim, num_roles)

        # 角色标签映射
        self.role_labels = {
            0: "O",           # Outside
            1: "B-ARG0",      # Agent (执行者)
            2: "I-ARG0",
            3: "B-ARG1",      # Patient (承受者)
            4: "I-ARG1",
            5: "B-ARGM-TMP",  # Time (时间)
            6: "I-ARGM-TMP",
            7: "B-ARGM-LOC",  # Location (地点)
            8: "I-ARGM-LOC",
            9: "B-ARGM-MNR",  # Manner (方式)
        }

    def forward(self, token_ids: torch.Tensor, predicate_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [batch, seq_len] token 索引
            predicate_positions: [batch, seq_len] 谓词位置指示 (0/1)
        Returns:
            [batch, seq_len, num_roles] 每个 token 的角色预测
        """
        # Embedding
        seq_len = token_ids.size(1)
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)

        token_emb = self.token_embedding(token_ids)
        pos_emb = self.position_embedding(positions)
        pred_emb = self.predicate_indicator(predicate_positions)

        # 融合
        x = token_emb + pos_emb + pred_emb

        # LSTM 编码
        h, _ = self.lstm(x)

        # 分类
        logits = self.classifier(h)
        return logits


def mock_tokenize(text: str, predicate: str) -> Tuple[List[str], int]:
    """简化的分词（演示用）"""
    tokens = text.split()
    pred_idx = tokens.index(predicate) if predicate in tokens else 2
    return tokens, pred_idx


def extract_srl_frames(sentence: str, model: BERTSRLModel) -> List[SRLFrame]:
    """
    从句子中提取 SRL 框架

    简化版：基于规则识别谓词和论元（生产环境使用训练好的 BERT-SRL）
    """
    frames = []

    # 谓词词典（演示用）
    predicates = {
        "买了": {"args": [("ARG0", "买家"), ("ARG1", "购买物")]},
        "用了": {"args": [("ARG0", "使用者"), ("ARG1", "使用物")]},
        "推荐": {"args": [("ARG0", "推荐者"), ("ARG1", "推荐物"), ("ARG2", "被推荐者")]},
        "觉得": {"args": [("ARG0", "感受者"), ("ARG1", "感受内容")]},
        "等": {"args": [("ARG0", "等待者"), ("ARGM-TMP", "时间")]},
        "bought": {"args": [("ARG0", "buyer"), ("ARG1", "item")]},
        "love": {"args": [("ARG0", "lover"), ("ARG1", "loved")]},
    }

    words = sentence.split()

    for i, word in enumerate(words):
        if word in predicates:
            pred_info = predicates[word]
            args = []

            # 简化的论元检测（基于位置启发式）
            if i > 0:
                args.append({"role": "ARG0", "text": words[i-1], "span": (i-1, i)})
            if i + 1 < len(words):
                args.append({"role": "ARG1", "text": words[i+1], "span": (i+1, i+2)})

            frames.append(SRLFrame(
                predicate=word,
                predicate_span=(i, i+1),
                arguments=args,
                sentence=sentence
            ))

    return frames


# ============================================
# 母婴电商评论 SRL 抽取示例
# ============================================

def demo_srl_extraction():
    """演示从母婴电商评论中抽取语义角色"""
    print("=" * 70)
    print("BERT-SRL — 语义角色标注")
    print("=" * 70)

    model = BERTSRLModel(vocab_size=1000, hidden_dim=128, num_roles=10)

    reviews = [
        "妈妈买了 Spectra S1 吸奶器",
        "宝宝用了 防胀气奶瓶 觉得很舒服",
        "朋友推荐 温奶器 给新手妈妈",
        "买家等了 一周 才收到储奶袋",
        "I bought the breast pump and love it",
    ]

    print("\n[抽取语义角色框架]\n")

    for review in reviews:
        print(f"句子: {review}")
        frames = extract_srl_frames(review, model)

        for frame in frames:
            print(f"  谓词: [{frame.predicate}] (位置: {frame.predicate_span})")
            for arg in frame.arguments:
                print(f"    → {arg['role']}: '{arg['text']}'")
        print()

    print("=" * 70)


def demonstrate_srl_to_event_frame():
    """演示 SRL 结果如何组装成事件框架"""
    print("\n" + "=" * 70)
    print("SRL → 事件框架 (Event Frame)")
    print("=" * 70)

    srl_results = [
        {
            "sentence": "妈妈买了 Spectra S1 吸奶器",
            "frames": [
                {
                    "predicate": "买了",
                    "event_type": "PURCHASE",
                    "arguments": {
                        "AGENT": "妈妈",
                        "THEME": "Spectra S1 吸奶器",
                    }
                }
            ]
        },
        {
            "sentence": "物流太慢了等了一周",
            "frames": [
                {
                    "predicate": "等",
                    "event_type": "WAIT",
                    "arguments": {
                        "AGENT": "买家",
                        "TIME": "一周",
                    }
                }
            ]
        },
    ]

    print("""
    事件框架组装规则:
      1. 谓词 → Event Type (通过映射表)
         "买了" → PURCHASE
         "等了" → WAIT
         "推荐" → RECOMMENDATION

      2. 论元 → Event Arguments
         ARG0 (Agent) → 事件执行者
         ARG1 (Patient) → 事件对象
         ARGM-TMP → 时间
         ARGM-LOC → 地点

      3. 事件图构建
         事件节点: (type, trigger, arguments)
         事件间关系: BEFORE, AFTER, CAUSE, ENABLE
    """)

    for result in srl_results:
        print(f"\n  句子: {result['sentence']}")
        for frame in result["frames"]:
            print(f"    事件: {frame['event_type']} (触发词: {frame['predicate']})")
            for role, value in frame["arguments"].items():
                print(f"      {role}: {value}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_srl_extraction()
    demonstrate_srl_to_event_frame()

    print("\n生产环境建议:")
    print("  1. 使用 transformers.BertModel 替代简化 embedding")
    print("  2. 在 CoNLL 2009/2012 上预训练，领域数据微调")
    print("  3. 结合依存句法树提升论元识别准确率")
    print("  4. 将 SRL 输出接入事件图构建模块")
