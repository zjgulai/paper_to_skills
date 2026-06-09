"""
Model Evaluation Metrics — ROC/AUC/PR/Calibration 模型评估体系
paper2skills-code: 12-ML基础 | 母婴出海跨境电商
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass


@dataclass
class EvalReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    auc_pr: float
    calibration_error: float  # Expected Calibration Error (ECE)
    business_value: float     # 业务价值（节省金额/提升转化）


def compute_confusion_matrix(probs: list[float], labels: list[int],
                              threshold: float = 0.5) -> dict:
    tp = sum(1 for p, l in zip(probs, labels) if p >= threshold and l == 1)
    fp = sum(1 for p, l in zip(probs, labels) if p >= threshold and l == 0)
    tn = sum(1 for p, l in zip(probs, labels) if p < threshold and l == 0)
    fn = sum(1 for p, l in zip(probs, labels) if p < threshold and l == 1)
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def auc_roc(probs: list[float], labels: list[int]) -> float:
    pos = [p for p, l in zip(probs, labels) if l == 1]
    neg = [p for p, l in zip(probs, labels) if l == 0]
    if not pos or not neg:
        return 0.5
    correct_pairs = sum(1 for p in pos for n in neg if p > n)
    total_pairs = len(pos) * len(neg)
    return round(correct_pairs / total_pairs, 4)


def auc_pr(probs: list[float], labels: list[int]) -> float:
    sorted_pairs = sorted(zip(probs, labels), reverse=True)
    precisions, recalls = [], []
    tp = fp = 0
    total_pos = sum(labels)
    for prob, label in sorted_pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / total_pos if total_pos > 0 else 0)

    aupr = sum((recalls[i] - recalls[i-1]) * precisions[i]
               for i in range(1, len(recalls))) if len(recalls) > 1 else 0
    return round(abs(aupr), 4)


def expected_calibration_error(probs: list[float], labels: list[int],
                                n_bins: int = 10) -> float:
    bin_size = 1.0 / n_bins
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = i * bin_size, (i + 1) * bin_size
        bin_indices = [j for j, p in enumerate(probs) if lo <= p < hi]
        if not bin_indices:
            continue
        avg_confidence = sum(probs[j] for j in bin_indices) / len(bin_indices)
        avg_accuracy = sum(labels[j] for j in bin_indices) / len(bin_indices)
        ece += len(bin_indices) / n * abs(avg_confidence - avg_accuracy)
    return round(ece, 4)


def evaluate_model(probs: list[float], labels: list[int],
                   revenue_per_tp: float = 500.0,
                   cost_per_fp: float = 50.0) -> EvalReport:
    cm = compute_confusion_matrix(probs, labels)
    tp, fp, fn = cm["tp"], cm["fp"], cm["fn"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)
    acc = (tp + cm["tn"]) / len(labels)
    business_value = tp * revenue_per_tp - fp * cost_per_fp

    return EvalReport(
        accuracy=round(acc, 4), precision=round(precision, 4),
        recall=round(recall, 4), f1=round(f1, 4),
        auc_roc=auc_roc(probs, labels),
        auc_pr=auc_pr(probs, labels),
        calibration_error=expected_calibration_error(probs, labels),
        business_value=round(business_value, 0),
    )


def run_evaluation_demo():
    random.seed(42)
    labels = [1] * 200 + [0] * 800
    probs = [random.betavariate(4, 2) if l == 1 else random.betavariate(2, 5)
             for l in labels]

    report = evaluate_model(probs, labels)
    print("=== 模型评估报告（母婴 Churn 预测）===")
    print(f"  Accuracy:     {report.accuracy:.4f}")
    print(f"  Precision:    {report.precision:.4f}")
    print(f"  Recall:       {report.recall:.4f}")
    print(f"  F1:           {report.f1:.4f}")
    print(f"  AUC-ROC:      {report.auc_roc:.4f}")
    print(f"  AUC-PR:       {report.auc_pr:.4f}")
    print(f"  ECE:          {report.calibration_error:.4f} (越小越好)")
    print(f"  业务价值:      ¥{report.business_value:,.0f}/月")
    print("\n✅ 模型评估演示完成")


if __name__ == "__main__":
    run_evaluation_demo()
