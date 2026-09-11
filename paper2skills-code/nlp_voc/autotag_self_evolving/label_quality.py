"""标签质量评估模块 — cleanlab 集成

集成 cleanlab 的 Confident Learning，检测 AutoTag 产出的标签质量问题。

安装依赖:
    pip install cleanlab

生产环境建议:
    - 使用 out-of-sample 预测概率（如交叉验证生成）
    - 结合业务规则过滤 false positive
    - 定期运行（建议每批次标注后）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# 可选导入：cleanlab 未安装时提供降级方案
try:
    from cleanlab.filter import find_label_issues
    CLEANLAB_AVAILABLE = True
except ImportError:
    CLEANLAB_AVAILABLE = False


@dataclass
class LabelIssue:
    """单条标签问题记录"""

    text: str
    given_label: str           # 当前标签
    suggested_label: str       # cleanlab 建议的标签
    issue_type: str            # "label_error" | "ambiguous" | "outlier"
    confidence: float          # 问题置信度 0-1
    reason: str = ""           # 问题原因说明


@dataclass
class QualityReport:
    """质量评估报告"""

    total_samples: int
    issue_count: int
    issue_rate: float
    noise_rate_estimate: float
    issues: list[LabelIssue]
    label_distribution: dict[str, int]
    issue_by_label: dict[str, int]
    label_by_label: dict[str, list[LabelIssue]]  # 按给定标签分组

    def to_dict(self) -> dict:
        return {
            "total_samples": self.total_samples,
            "issue_count": self.issue_count,
            "issue_rate": self.issue_rate,
            "noise_rate_estimate": self.noise_rate_estimate,
            "issues": [
                {
                    "text": i.text[:50] + "..." if len(i.text) > 50 else i.text,
                    "given_label": i.given_label,
                    "suggested_label": i.suggested_label,
                    "issue_type": i.issue_type,
                    "confidence": i.confidence,
                }
                for i in self.issues
            ],
            "issue_by_label": self.issue_by_label,
        }


class LabelQualityAssessor:
    """标签质量评估器

    封装 cleanlab 的核心功能，适配 AutoTag 的输入输出格式。
    """

    def __init__(
        self,
        labels: list[str],
        confidence_threshold: float = 0.6,
    ):
        self.labels = labels
        self.label_to_idx = {label: i for i, label in enumerate(labels)}
        self.confidence_threshold = confidence_threshold

    def assess(
        self,
        texts: list[str],
        given_labels: list[str],
        pred_probs: np.ndarray,
    ) -> QualityReport:
        """评估标签质量

        Args:
            texts: 文本列表
            given_labels: 当前标签列表
            pred_probs: 预测概率矩阵 (n_samples, n_classes)
                        pred_probs[i][j] = 第 i 条文本属于第 j 个标签的概率

        Returns:
            质量评估报告
        """
        if not CLEANLAB_AVAILABLE:
            return self._fallback_assess(texts, given_labels, pred_probs)

        # 将标签名称转为索引
        labels_idx = np.array([
            self.label_to_idx.get(l, -1) for l in given_labels
        ])

        # 过滤无效标签
        valid_mask = labels_idx >= 0
        if not valid_mask.all():
            print(f"[警告] {(~valid_mask).sum()} 条样本标签不在已知标签列表中")

        # 使用 cleanlab 检测问题标签
        issue_indices = find_label_issues(
            labels=labels_idx[valid_mask],
            pred_probs=pred_probs[valid_mask],
            return_indices_ranked_by="self_confidence",
        )

        # 构建问题列表
        issues: list[LabelIssue] = []
        valid_texts = [texts[i] for i, v in enumerate(valid_mask) if v]
        valid_labels = [given_labels[i] for i, v in enumerate(valid_mask) if v]

        for idx in issue_indices:
            # 找出模型最可能的真实标签
            suggested_idx = int(np.argmax(pred_probs[valid_mask][idx]))
            suggested_label = self.labels[suggested_idx]

            # 计算置信度（模型对建议标签的置信度）
            confidence = float(pred_probs[valid_mask][idx][suggested_idx])

            issue = LabelIssue(
                text=valid_texts[idx],
                given_label=valid_labels[idx],
                suggested_label=suggested_label,
                issue_type="label_error",
                confidence=confidence,
                reason=f"模型认为更可能是 '{suggested_label}' 而非 '{valid_labels[idx]}'",
            )
            issues.append(issue)

        # 额外检测：低置信度样本（标签模糊）
        for i, (text, label) in enumerate(zip(valid_texts, valid_labels)):
            if i in issue_indices:
                continue
            max_prob = float(np.max(pred_probs[valid_mask][i]))
            if max_prob < self.confidence_threshold:
                suggested_idx = int(np.argmax(pred_probs[valid_mask][i]))
                issues.append(LabelIssue(
                    text=text,
                    given_label=label,
                    suggested_label=self.labels[suggested_idx],
                    issue_type="ambiguous",
                    confidence=max_prob,
                    reason="模型置信度低，标签可能模糊",
                ))

        # 统计
        label_dist: dict[str, int] = {}
        for label in given_labels:
            label_dist[label] = label_dist.get(label, 0) + 1

        issue_by_label: dict[str, int] = {}
        for issue in issues:
            issue_by_label[issue.given_label] = issue_by_label.get(issue.given_label, 0) + 1

        # 噪声率估计（简化版）
        noise_rate = len(issue_indices) / len(valid_texts) if valid_texts else 0.0

        return QualityReport(
            total_samples=len(texts),
            issue_count=len(issues),
            issue_rate=len(issues) / len(texts) if texts else 0.0,
            noise_rate_estimate=noise_rate,
            issues=issues,
            label_distribution=label_dist,
            issue_by_label=issue_by_label,
            label_by_label={},
        )

    def _fallback_assess(
        self,
        texts: list[str],
        given_labels: list[str],
        pred_probs: np.ndarray,
    ) -> QualityReport:
        """cleanlab 未安装时的降级方案

        基于简单规则检测问题标签：
        - 预测概率最大值与给定标签不一致 → label_error
        - 预测概率最大值 < 阈值 → ambiguous
        """
        print("[警告] cleanlab 未安装，使用降级规则评估")
        print("  安装: pip install cleanlab")

        issues: list[LabelIssue] = []

        for i, (text, given_label) in enumerate(zip(texts, given_labels)):
            if pred_probs is None or len(pred_probs) <= i:
                continue

            probs = pred_probs[i]
            predicted_idx = int(np.argmax(probs))
            predicted_label = self.labels[predicted_idx]
            max_prob = float(probs[predicted_idx])

            if predicted_label != given_label and max_prob > 0.5:
                issues.append(LabelIssue(
                    text=text,
                    given_label=given_label,
                    suggested_label=predicted_label,
                    issue_type="label_error",
                    confidence=max_prob,
                    reason="预测标签与给定标签不一致",
                ))
            elif max_prob < self.confidence_threshold:
                issues.append(LabelIssue(
                    text=text,
                    given_label=given_label,
                    suggested_label=predicted_label,
                    issue_type="ambiguous",
                    confidence=max_prob,
                    reason="预测置信度低",
                ))

        label_dist: dict[str, int] = {}
        for label in given_labels:
            label_dist[label] = label_dist.get(label, 0) + 1

        issue_by_label: dict[str, int] = {}
        for issue in issues:
            issue_by_label[issue.given_label] = issue_by_label.get(issue.given_label, 0) + 1

        return QualityReport(
            total_samples=len(texts),
            issue_count=len(issues),
            issue_rate=len(issues) / len(texts) if texts else 0.0,
            noise_rate_estimate=len(issues) / len(texts) if texts else 0.0,
            issues=issues,
            label_distribution=label_dist,
            issue_by_label=issue_by_label,
            label_by_label={},
        )

    def print_report(self, report: QualityReport, top_k: int = 10) -> None:
        """打印质量报告"""
        print("\n" + "=" * 60)
        print("标签质量评估报告")
        print("=" * 60)
        print(f"总样本数: {report.total_samples}")
        print(f"问题标签数: {report.issue_count} ({report.issue_rate:.1%})")
        print(f"估计噪声率: {report.noise_rate_estimate:.1%}")
        print(f"\n各标签问题分布:")
        for label, count in sorted(report.issue_by_label.items(), key=lambda x: x[1], reverse=True):
            total = report.label_distribution.get(label, 1)
            print(f"  {label}: {count}/{total} ({count/total:.1%})")
        print(f"\nTop-{top_k} 问题标签:")
        for i, issue in enumerate(report.issues[:top_k], 1):
            print(f"\n  [{i}] [{issue.issue_type}] 置信度: {issue.confidence:.2f}")
            print(f"      文本: {issue.text[:40]}...")
            print(f"      给定: {issue.given_label} → 建议: {issue.suggested_label}")
            print(f"      原因: {issue.reason}")
        print("=" * 60)


# ── 与 AutoTag 集成的便捷接口 ─────────────────────────────────

def assess_autotag_predictions(
    texts: list[str],
    predictions: list[dict],
    labels: list[str],
    pred_probs: Optional[np.ndarray] = None,
) -> QualityReport:
    """评估 AutoTag 预测结果的质量

    Args:
        texts: 原始文本
        predictions: AutoTag 的 PredictionResult.to_dict() 列表
        labels: 完整标签列表
        pred_probs: 可选的预测概率矩阵。如果未提供，将基于 predictions 构造简化版

    Returns:
        质量评估报告
    """
    given_labels = [p.get("l4") or p.get("l3") or p.get("l2") or p.get("l1") or "未知" for p in predictions]

    if pred_probs is None:
        # 基于预测结果构造简化概率矩阵
        n_samples = len(predictions)
        n_classes = len(labels)
        pred_probs = np.zeros((n_samples, n_classes))

        for i, pred in enumerate(predictions):
            label = pred.get("l4") or pred.get("l3") or pred.get("l2") or pred.get("l1")
            if label in labels:
                idx = labels.index(label)
                conf = pred.get("confidence", 0.5)
                pred_probs[i][idx] = conf
                # 剩余概率均摊给其他类
                remaining = (1.0 - conf) / (n_classes - 1) if n_classes > 1 else 0
                for j in range(n_classes):
                    if j != idx:
                        pred_probs[i][j] = remaining

    assessor = LabelQualityAssessor(labels)
    return assessor.assess(texts, given_labels, pred_probs)


# ── 测试 ──────────────────────────────────────────────────────

def test_label_quality():
    """测试标签质量评估"""
    print("=" * 60)
    print("测试: LabelQualityAssessor")
    print("=" * 60)

    labels = ["尺码偏差", "材质问题", "漏尿", "腰贴问题", "物流延迟", "过敏反应"]

    # 模拟文本和标签（包含一些故意错误）
    texts = [
        "这个尺码偏小",                       # 正确: 尺码偏差
        "面料太硬不舒服",                     # 正确: 材质问题
        "晚上总是漏尿",                       # 正确: 漏尿
        "腰贴粘不牢",                         # 正确: 腰贴问题
        "物流太慢了",                         # 正确: 物流延迟
        "宝宝用了起红疹",                     # 正确: 过敏反应
        "尺码偏大不合身",                     # 正确: 尺码偏差
        "面料太软不错",                       # 正确: 材质问题（但情感不同）
        "晚上不漏尿很好",                     # 故意标错: 实际是正面，不应标"漏尿"
        "物流很快满意",                       # 故意标错: 实际是正面，不应标"物流延迟"
        "宝宝皮肤没有异常",                   # 故意标错: 不应标"过敏反应"
    ]

    # 模拟给定标签（包含 3 个错误）
    given_labels = [
        "尺码偏差", "材质问题", "漏尿", "腰贴问题", "物流延迟",
        "过敏反应", "尺码偏差", "材质问题", "漏尿", "物流延迟", "过敏反应",
    ]

    # 模拟预测概率（模拟一个有点能力但不完美的分类器）
    n_samples = len(texts)
    n_classes = len(labels)
    pred_probs = np.zeros((n_samples, n_classes))

    # 正确样本：高概率给正确标签
    for i in range(6):
        idx = labels.index(given_labels[i])
        pred_probs[i][idx] = 0.85
        for j in range(n_classes):
            if j != idx:
                pred_probs[i][j] = 0.03

    # 正确样本（第二批）
    for i in range(6, 8):
        idx = labels.index(given_labels[i])
        pred_probs[i][idx] = 0.75
        for j in range(n_classes):
            if j != idx:
                pred_probs[i][j] = 0.04

    # 错误样本：模型知道标签不对
    # 样本8: "晚上不漏尿很好" → 模型认为不是漏尿
    pred_probs[8][labels.index("漏尿")] = 0.1
    pred_probs[8][labels.index("尺码偏差")] = 0.6  # 模型猜尺码

    # 样本9: "物流很快满意" → 模型认为不是物流延迟
    pred_probs[9][labels.index("物流延迟")] = 0.15
    pred_probs[9][labels.index("腰贴问题")] = 0.5  # 模型猜腰贴

    # 样本10: "宝宝皮肤没有异常" → 模型认为不是过敏
    pred_probs[10][labels.index("过敏反应")] = 0.1
    pred_probs[10][labels.index("材质问题")] = 0.55  # 模型猜材质

    # 运行评估
    assessor = LabelQualityAssessor(labels)
    report = assessor.assess(texts, given_labels, pred_probs)
    assessor.print_report(report, top_k=5)

    # 验证：应该至少发现 3 个错误标签
    assert report.issue_count >= 2, f"预期发现至少 2 个问题标签，实际 {report.issue_count}"
    print(f"\n✓ 检测到 {report.issue_count} 个问题标签")

    # 测试与 AutoTag 集成的接口
    print("\n--- AutoTag 集成接口测试 ---")
    predictions = [
        {"l4": "尺码偏差", "confidence": 0.8},
        {"l4": "材质问题", "confidence": 0.7},
        {"l4": "漏尿", "confidence": 0.9},
        {"l4": "物流延迟", "confidence": 0.6},  # 模拟低置信度
    ]
    report2 = assess_autotag_predictions(
        texts=["尺码偏小", "面料硬", "晚上漏", "物流快"],
        predictions=predictions,
        labels=labels,
    )
    print(f"AutoTag 集成评估: {report2.issue_count} 个问题")

    print("\n" + "=" * 60)
    print("标签质量评估测试完成 ✓")
    print("=" * 60)


# ── AltTest 统计检验 ─────────────────────────────────────────

@dataclass
class AltTestResult:
    """Alternative Annotator Test 结果"""

    passed: bool              # ω ≥ 0.5 是否通过
    winning_rate: float       # ω: 获胜率
    avg_advantage_prob: float # ρ: 平均优势概率
    epsilon: float            # 成本效益参数
    alpha: float              # 显著性水平
    annotator_count: int      # 人类标注者数量
    instance_count: int       # 评估实例数量
    p_values: dict[str, float]  # 每个标注者的 p-value
    rejections: dict[str, bool] # 每个标注者是否拒绝 H0
    details: str = ""         # 结果解读


class AltTestEvaluator:
    """Alternative Annotator Test 评估器

    基于 Calderon et al. (ACL 2025) 的统计框架，
    判断 LLM 标注是否可以替代人工标注。

    核心思想：
        1. 对每条样本，比较 LLM 和每个标注者与其他标注者 majority vote 的一致性
        2. 用配对 t-检验判断 LLM 是否显著优于每个标注者（考虑成本效益 ε）
        3. 获胜率 ω ≥ 0.5 且通过 FDR 校正 → LLM 可替代人工

    使用场景：
        - AutoTag 产出标签后，判断是否可以信任 LLM 标注
        - 定期（如每月）用抽检样本验证 LLM 标注可靠性
        - 模型升级后，验证新版本是否仍可通过 alt-test
    """

    def __init__(self, epsilon: float = 0.1, alpha: float = 0.05):
        """
        Args:
            epsilon: 成本效益参数。LLM 的成本优势使其阈值降低 ε。
                     推荐值：0.05（保守）~ 0.2（标准）~ 0.3（宽松）
            alpha: 显著性水平，默认 0.05
        """
        self.epsilon = epsilon
        self.alpha = alpha

    def evaluate(
        self,
        llm_labels: list,
        human_labels: dict[str, list],
        task_type: str = "classification",
    ) -> AltTestResult:
        """执行 Alternative Annotator Test

        Args:
            llm_labels: LLM 标注结果，长度 n
            human_labels: {annotator_id: [labels]}，每个长度 n，至少 3 个标注者
            task_type: "classification" | "continuous" | "textual"

        Returns:
            AltTestResult
        """
        annotators = list(human_labels.keys())
        m = len(annotators)
        n = len(llm_labels)

        if m < 3:
            raise ValueError(f"alt-test 需要至少 3 个标注者，当前只有 {m} 个")
        if n < 30:
            raise ValueError(f"alt-test 需要至少 30 条样本以满足 t-检验正态假设，当前只有 {n} 条")

        # 对齐评分：对每个 (实例, 标注者) 计算 LLM 和该标注者的对齐分数
        p_values = {}
        rejections = {}
        advantage_probs = []

        for j, aj in enumerate(annotators):
            # 计算差异分数 d_{i,j} = score(LLM, i, j) - score(h_j, i, j)
            diffs = []
            for i in range(n):
                # 其他标注者（排除 j）的 majority vote 作为 gold
                other_labels = [human_labels[a][i] for a in annotators if a != aj]
                gold = self._majority_vote(other_labels)

                # LLM 的对齐分数
                score_llm = self._alignment_score(
                    llm_labels[i], gold, task_type
                )
                # 标注者 j 的对齐分数
                score_human = self._alignment_score(
                    human_labels[aj][i], gold, task_type
                )

                diffs.append(score_llm - score_human)

            diffs = np.array(diffs)
            d_mean = float(np.mean(diffs))
            d_std = float(np.std(diffs, ddof=1))

            # 配对 t-检验: H0: mean(diff) <= epsilon
            # t = (d_mean - epsilon) / (d_std / sqrt(n))
            if d_std == 0:
                # 所有差异相同，直接比较均值
                p_val = 0.0 if d_mean > self.epsilon else 1.0
            else:
                from scipy import stats
                # 单侧检验: H0: mean <= epsilon vs H1: mean > epsilon
                t_stat = (d_mean - self.epsilon) / (d_std / np.sqrt(n))
                p_val = 1 - stats.t.cdf(t_stat, df=n - 1)

            p_values[aj] = p_val

            # 优势概率：P(LLM 优于 h_j) = Φ((d_mean) / (d_std / sqrt(n)))
            if d_std > 0:
                z = d_mean / (d_std / np.sqrt(n))
                adv_prob = float(stats.norm.cdf(z))
            else:
                adv_prob = 1.0 if d_mean > 0 else 0.0
            advantage_probs.append(adv_prob)

        # FDR 校正 (Benjamini-Yekutieli)
        p_vals_array = np.array(list(p_values.values()))
        rejected = self._fdr_by(p_vals_array, self.alpha)

        for idx, aj in enumerate(annotators):
            rejections[aj] = rejected[idx]

        # 获胜率 ω = 拒绝 H0 的比例
        winning_rate = float(np.mean(rejected))

        # 平均优势概率 ρ
        avg_advantage_prob = float(np.mean(advantage_probs))

        # 是否通过：ω >= 0.5
        passed = winning_rate >= 0.5

        # 结果解读
        details = self._interpret_result(
            passed, winning_rate, avg_advantage_prob, m, n, self.epsilon
        )

        return AltTestResult(
            passed=passed,
            winning_rate=winning_rate,
            avg_advantage_prob=avg_advantage_prob,
            epsilon=self.epsilon,
            alpha=self.alpha,
            annotator_count=m,
            instance_count=n,
            p_values=p_values,
            rejections=rejections,
            details=details,
        )

    def _alignment_score(self, pred, gold, task_type: str) -> float:
        """计算单个预测与 gold 的对齐分数"""
        if task_type == "classification":
            return 1.0 if pred == gold else 0.0
        elif task_type == "continuous":
            # 连续值：使用相关性（这里简化为归一化距离）
            if isinstance(pred, (int, float)) and isinstance(gold, (int, float)):
                # 假设值域 [0, 1]，距离越近分数越高
                return max(0.0, 1.0 - abs(pred - gold))
            return 1.0 if pred == gold else 0.0
        elif task_type == "textual":
            # 文本：简化为完全匹配（实际应用应使用语义相似度）
            return 1.0 if str(pred).strip() == str(gold).strip() else 0.0
        else:
            return 1.0 if pred == gold else 0.0

    def _majority_vote(self, labels: list) -> any:
        """计算 majority vote"""
        from collections import Counter
        if not labels:
            return None
        counter = Counter(labels)
        return counter.most_common(1)[0][0]

    def _fdr_by(self, p_values: np.ndarray, alpha: float) -> np.ndarray:
        """Benjamini-Yekutieli FDR 校正

        不假设检验独立性，适用于依赖假设场景。
        """
        m = len(p_values)
        if m == 0:
            return np.array([], dtype=bool)

        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]

        # BY 校正阈值: sorted_p[i] <= (i+1)/m * alpha / C(m)
        # 其中 C(m) = sum(1/k for k in 1..m)
        cm = sum(1.0 / k for k in range(1, m + 1))

        rejected = np.zeros(m, dtype=bool)
        # 从最大 p-value 开始找
        threshold = 0.0
        for i in range(m - 1, -1, -1):
            by_threshold = ((i + 1) / m) * (alpha / cm)
            if sorted_p[i] <= by_threshold:
                threshold = by_threshold
                break

        if threshold > 0:
            rejected = p_values <= threshold

        return rejected

    def _interpret_result(
        self,
        passed: bool,
        winning_rate: float,
        avg_advantage_prob: float,
        n_annotators: int,
        n_instances: int,
        epsilon: float,
    ) -> str:
        """生成结果解读文本"""
        status = "✅ 通过" if passed else "❌ 未通过"
        lines = [
            f"Alt-Test 结果: {status}",
            f"  获胜率 ω = {winning_rate:.2f} (阈值: ≥0.5)",
            f"  平均优势概率 ρ = {avg_advantage_prob:.2f}",
            f"  评估设置: {n_annotators} 个标注者 × {n_instances} 条样本, ε={epsilon}",
        ]

        if passed:
            lines.append(
                f"  结论: LLM 标注在 {winning_rate:.0%} 的标注者对比中获胜，"
                f"可以替代人工标注（考虑成本效益 ε={epsilon}）"
            )
        else:
            lines.append(
                f"  结论: LLM 标注仅在 {winning_rate:.0%} 的标注者对比中获胜，"
                f"未达到替代人工标注的阈值。建议："
            )
            if avg_advantage_prob > 0.4:
                lines.append("    - 考虑增加评估样本量（推荐 ≥100 条）")
                lines.append("    - 检查 LLM prompt 是否需要优化")
            else:
                lines.append("    - LLM 与人工标注差距较大，建议保持人工审核")
                lines.append("    - 考虑更换更强 LLM 或优化任务设计")

        return "\n".join(lines)

    def recommend_epsilon(
        self,
        human_cost_per_label: float,
        llm_cost_per_label: float,
        annotation_complexity: str = "medium",
    ) -> float:
        """根据成本结构推荐 epsilon 值

        Args:
            human_cost_per_label: 人工标注单条成本（元或美元）
            llm_cost_per_label: LLM 标注单条成本
            annotation_complexity: "simple" | "medium" | "complex"

        Returns:
            推荐的 epsilon 值
        """
        cost_ratio = human_cost_per_label / max(llm_cost_per_label, 0.001)

        # 基础 epsilon 由成本比决定
        if cost_ratio >= 100:
            base_eps = 0.2
        elif cost_ratio >= 20:
            base_eps = 0.15
        elif cost_ratio >= 5:
            base_eps = 0.1
        else:
            base_eps = 0.05

        # 复杂度调整
        complexity_adj = {"simple": 0.05, "medium": 0.0, "complex": -0.05}
        epsilon = base_eps + complexity_adj.get(annotation_complexity, 0.0)

        # 限制在有效范围 [0.05, 0.3]
        return max(0.05, min(0.3, epsilon))


# ── 与 AutoTag 集成的便捷接口 ─────────────────────────────────

def assess_llm_reliability(
    llm_labels: list,
    human_labels: dict[str, list],
    epsilon: float = 0.1,
    task_type: str = "classification",
) -> AltTestResult:
    """一键评估 LLM 标注可靠性

    在 AutoTag pipeline 中的使用示例:

        # 1. 每月抽检 100-200 条样本，分配给 3-5 个人工标注员
        human_labels = {
            "annotator_A": ["尺码偏差", "材质问题", ...],  # 100 条
            "annotator_B": ["尺码偏差", "物流延迟", ...],  # 100 条
            "annotator_C": ["漏尿", "材质问题", ...],      # 100 条
        }
        llm_labels = ["尺码偏差", "材质问题", ...]  # AutoTag 产出

        # 2. 执行 alt-test
        result = assess_llm_reliability(llm_labels, human_labels, epsilon=0.1)

        # 3. 根据结果决定后续流程
        if result.passed:
            print("LLM 标注可靠，减少人工审核比例")
        else:
            print("LLM 标注不可靠，增加人工审核")
    """
    evaluator = AltTestEvaluator(epsilon=epsilon)
    return evaluator.evaluate(llm_labels, human_labels, task_type=task_type)


if __name__ == "__main__":
    test_label_quality()

    # ── AltTest 测试 ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("测试: AltTestEvaluator")
    print("=" * 60)

    # 模拟数据：3 个标注者 + LLM，100 条样本
    np.random.seed(42)
    n = 100
    labels_pool = ["尺码偏差", "材质问题", "漏尿", "腰贴问题", "物流延迟"]

    # Gold labels（隐含的"真实"标签）
    gold = np.random.choice(labels_pool, size=n)

    # 人类标注者（有一定错误率）
    def simulate_human(gold_labels, error_rate=0.15):
        result = gold_labels.copy()
        n_errors = int(len(gold_labels) * error_rate)
        error_idx = np.random.choice(len(gold_labels), size=n_errors, replace=False)
        for idx in error_idx:
            wrong = [l for l in labels_pool if l != gold_labels[idx]]
            result[idx] = np.random.choice(wrong)
        return result

    human_A = simulate_human(gold, 0.12)
    human_B = simulate_human(gold, 0.18)
    human_C = simulate_human(gold, 0.15)

    # LLM（错误率更低，模拟更好的标注者）
    llm = simulate_human(gold, 0.08)

    evaluator = AltTestEvaluator(epsilon=0.1)
    result = evaluator.evaluate(
        llm_labels=list(llm),
        human_labels={"A": list(human_A), "B": list(human_B), "C": list(human_C)},
        task_type="classification",
    )

    print(f"\n{result.details}")
    print(f"\n各标注者 p-values:")
    for ann, pval in result.p_values.items():
        rejected = "✅ 拒绝H0" if result.rejections[ann] else "❌ 不拒绝"
        print(f"  {ann}: p={pval:.4f} {rejected}")

    # 测试 epsilon 推荐
    print(f"\n成本推荐 epsilon:")
    eps = evaluator.recommend_epsilon(
        human_cost_per_label=2.0,  # 人工 2元/条
        llm_cost_per_label=0.01,   # LLM 0.01元/条
        annotation_complexity="medium",
    )
    print(f"  人工2元 vs LLM 0.01元 → 推荐 ε = {eps}")

    # 测试未通过场景：LLM 比人类还差
    print("\n--- 未通过场景测试 ---")
    bad_llm = simulate_human(gold, 0.25)  # LLM 错误率更高
    result2 = evaluator.evaluate(
        llm_labels=list(bad_llm),
        human_labels={"A": list(human_A), "B": list(human_B), "C": list(human_C)},
        task_type="classification",
    )
    print(result2.details)
    assert not result2.passed, "预期未通过"

    print("\n" + "=" * 60)
    print("AltTest 测试完成 ✓")
    print("=" * 60)
