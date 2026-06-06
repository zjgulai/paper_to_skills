"""
Privacy-Preserving Federated Collection - SF-UBM + MFG-RegretNet
arXiv: 2604.14833 / 2603.28329
"""

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class LocalDataShard:
    party_id: str
    platform: str
    n_samples: int
    local_gradients: List[float] = field(default_factory=list)
    privacy_budget: float = 1.0
    noise_scale: float = 0.0


@dataclass
class FederatedModel:
    weights: List[float]
    round_num: int = 0
    participating_parties: List[str] = field(default_factory=list)


class DifferentialPrivacyMechanism:
    """Adaptive Weighted FL: 分层差分隐私保护"""

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta

    def gaussian_noise_scale(self, sensitivity: float, n_samples: int) -> float:
        return sensitivity * math.sqrt(2 * math.log(1.25 / max(self.delta, 1e-10))) / (self.epsilon * math.sqrt(n_samples))

    def privatize_gradients(self, gradients: List[float], sensitivity: float = 1.0) -> List[float]:
        n = max(len(gradients), 1)
        noise_scale = self.gaussian_noise_scale(sensitivity, n)
        return [g + random.gauss(0, noise_scale) for g in gradients]

    def privacy_amplification(self, base_epsilon: float, sampling_rate: float) -> float:
        return base_epsilon * sampling_rate


class AdaptiveWeightedAggregator:
    """SF-UBM 风格：动态多维权重聚合跨平台梯度"""

    def compute_weights(self, shards: List[LocalDataShard]) -> List[float]:
        raw_weights = []
        for shard in shards:
            quality_factor = min(1.0, shard.n_samples / 1000)
            privacy_factor = shard.privacy_budget
            combined = quality_factor * privacy_factor
            raw_weights.append(combined)
        total = sum(raw_weights)
        if total == 0:
            return [1.0 / len(shards)] * len(shards)
        return [w / total for w in raw_weights]

    def aggregate(self, shards: List[LocalDataShard],
                  dp_mechanism: DifferentialPrivacyMechanism) -> List[float]:
        if not shards or not shards[0].local_gradients:
            return []
        weights = self.compute_weights(shards)
        n_params = len(shards[0].local_gradients)
        aggregated = [0.0] * n_params
        for shard, weight in zip(shards, weights):
            private_grads = dp_mechanism.privatize_gradients(shard.local_gradients)
            for i in range(min(n_params, len(private_grads))):
                aggregated[i] += weight * private_grads[i]
        return aggregated


class MIADefenseEvaluator:
    """模型反演攻击（MIA）防御评估"""

    def simulate_mia_success_rate(self, model: FederatedModel,
                                  noise_scale: float) -> float:
        base_rate = 0.5
        n_params = len(model.weights)
        complexity_factor = min(1.0, n_params / 100)
        noise_protection = max(0.0, 1.0 - noise_scale * 2)
        parties_factor = max(0.0, 1.0 - len(model.participating_parties) * 0.05)
        mia_rate = base_rate * complexity_factor * noise_protection * parties_factor
        return min(0.5, max(0.0, mia_rate))


class FederatedCollectionPipeline:
    def __init__(self, epsilon: float = 0.1, delta: float = 1e-5):
        self.dp = DifferentialPrivacyMechanism(epsilon, delta)
        self.aggregator = AdaptiveWeightedAggregator()
        self.mia_evaluator = MIADefenseEvaluator()
        self.model = FederatedModel(weights=[], round_num=0)

    def federated_round(self, shards: List[LocalDataShard]) -> Dict[str, Any]:
        aggregated = self.aggregator.aggregate(shards, self.dp)
        self.model.weights = aggregated
        self.model.round_num += 1
        self.model.participating_parties = [s.party_id for s in shards]
        avg_noise = sum(
            self.dp.gaussian_noise_scale(1.0, max(s.n_samples, 1)) for s in shards
        ) / max(len(shards), 1)
        mia_rate = self.mia_evaluator.simulate_mia_success_rate(self.model, avg_noise)
        return {
            "round": self.model.round_num,
            "parties": len(shards),
            "model_params": len(aggregated),
            "avg_noise_scale": avg_noise,
            "mia_success_rate": mia_rate,
            "privacy_amplified_epsilon": self.dp.privacy_amplification(
                self.dp.epsilon, len(shards) / 10
            ),
        }


def test_dp_noise_injection():
    dp = DifferentialPrivacyMechanism(epsilon=1.0, delta=1e-5)
    gradients = [0.5, -0.3, 0.8, -0.1, 0.4]
    private_grads = dp.privatize_gradients(gradients, sensitivity=1.0)
    assert len(private_grads) == len(gradients)
    assert private_grads != gradients
    noise_scale = dp.gaussian_noise_scale(1.0, 1000)
    assert noise_scale > 0
    print(f"[PASS] dp_noise: noise_scale={noise_scale:.4f}, grads_changed={private_grads[:2]}")


def test_adaptive_weighting():
    aggregator = AdaptiveWeightedAggregator()
    shards = [
        LocalDataShard("amazon", "amazon", 5000, [0.5, -0.3, 0.8], privacy_budget=0.9),
        LocalDataShard("tiktok", "tiktok", 1000, [0.4, -0.2, 0.7], privacy_budget=0.7),
        LocalDataShard("indie", "independent", 200, [0.6, -0.4, 0.9], privacy_budget=0.5),
    ]
    weights = aggregator.compute_weights(shards)
    assert abs(sum(weights) - 1.0) < 1e-6
    assert weights[0] > weights[1] > weights[2]
    print(f"[PASS] adaptive_weights: {[f'{w:.3f}' for w in weights]} (amazon > tiktok > indie)")


def test_mia_defense():
    evaluator = MIADefenseEvaluator()
    model = FederatedModel(weights=[0.1] * 10, participating_parties=["p1", "p2", "p3"])
    high_noise_rate = evaluator.simulate_mia_success_rate(model, noise_scale=2.0)
    low_noise_rate = evaluator.simulate_mia_success_rate(model, noise_scale=0.1)
    assert high_noise_rate < low_noise_rate
    assert high_noise_rate < 0.1
    print(f"[PASS] mia_defense: high_noise={high_noise_rate:.4f}, low_noise={low_noise_rate:.4f}")


def test_federated_pipeline():
    random.seed(42)
    pipeline = FederatedCollectionPipeline(epsilon=0.1)
    shards = [
        LocalDataShard("amazon", "amazon", 5000, [random.gauss(0, 0.1) for _ in range(10)]),
        LocalDataShard("tiktok", "tiktok", 2000, [random.gauss(0, 0.1) for _ in range(10)]),
        LocalDataShard("indie", "independent", 500, [random.gauss(0, 0.1) for _ in range(10)]),
    ]
    result = pipeline.federated_round(shards)
    assert result["parties"] == 3
    assert result["model_params"] == 10
    assert result["mia_success_rate"] < 0.2
    print(f"[PASS] federated_pipeline: round={result['round']}, "
          f"MIA_rate={result['mia_success_rate']:.4f}, noise={result['avg_noise_scale']:.4f}")


if __name__ == "__main__":
    test_dp_noise_injection()
    test_adaptive_weighting()
    test_mia_defense()
    test_federated_pipeline()
    print("\n✅ All tests passed")
