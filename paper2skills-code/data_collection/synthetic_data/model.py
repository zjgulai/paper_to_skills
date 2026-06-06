"""
Synthetic Data for E-commerce - SIGIR'26 + ICML'26 + SCALR
arXiv: 2602.23620 / 2602.07298 / 2606.00282
"""

import random
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ProductQuery:
    query_text: str
    category: str
    is_long_tail: bool = False
    target_sku: str = ""


@dataclass
class SyntheticSample:
    query: str
    item_id: str
    label: float
    source: str = "synthetic"
    domain: str = ""


class LongTailQueryRewriter:
    """SIGIR'26: 多奖励 RL Query Rewriting 解决长尾知识密集型查询"""

    EXPANSION_TEMPLATES = {
        "pump": ["electric {}", "wearable {} pump", "{} for nursing mothers", "double {} breast"],
        "monitor": ["smart baby {}", "HD {}", "{} with night vision", "wifi baby {}"],
        "sterilizer": ["uv-c {}", "electric bottle {}", "{} warmer", "microwave {}"],
    }

    def rewrite(self, query: ProductQuery, n_variants: int = 3) -> List[str]:
        variants = [query.query_text]
        category_lower = query.category.lower()
        for key, templates in self.EXPANSION_TEMPLATES.items():
            if key in category_lower or key in query.query_text.lower():
                for tmpl in templates[:n_variants]:
                    variants.append(tmpl.format(query.query_text))
                break
        if query.is_long_tail:
            variants.append(f"best {query.query_text} for newborn")
            variants.append(f"{query.query_text} review comparison")
        return list(dict.fromkeys(variants))[:n_variants + 1]


class CrossDomainSynthesizer:
    """SCALR: 跨域事件迁移合成数据 — 用源域行为生成目标域样本"""

    def __init__(self, transfer_noise: float = 0.1):
        self.noise = transfer_noise

    def synthesize_from_source(
        self,
        source_samples: List[SyntheticSample],
        target_domain: str,
        n_samples: int = 100,
    ) -> List[SyntheticSample]:
        if not source_samples:
            return []
        synthetic = []
        for i in range(n_samples):
            base = random.choice(source_samples)
            noisy_label = base.label + random.gauss(0, self.noise)
            noisy_label = max(0.0, min(1.0, noisy_label))
            query_parts = base.query.split()
            if len(query_parts) > 1:
                idx = random.randint(0, len(query_parts) - 1)
                query_parts[idx] = query_parts[idx] + f"_{target_domain[:3]}"
            synthetic.append(SyntheticSample(
                query=" ".join(query_parts),
                item_id=f"syn_{target_domain}_{i:04d}",
                label=noisy_label,
                source="cross_domain_transfer",
                domain=target_domain,
            ))
        return synthetic


class ScalingLawValidator:
    """ICML'26: 验证合成数据是否满足 Scaling Law（recall 随数据量幂律增长）"""

    def __init__(self, min_r_squared: float = 0.85):
        self.min_r2 = min_r_squared

    def fit_power_law(self, data_sizes: List[int], recalls: List[float]) -> Tuple[float, float, float]:
        if len(data_sizes) < 3:
            return 0.0, 0.0, 0.0
        log_x = [math.log(max(n, 1)) for n in data_sizes]
        log_y = [math.log(max(r, 1e-6)) for r in recalls]
        n = len(log_x)
        mean_x = sum(log_x) / n
        mean_y = sum(log_y) / n
        ss_xy = sum((log_x[i] - mean_x) * (log_y[i] - mean_y) for i in range(n))
        ss_xx = sum((log_x[i] - mean_x) ** 2 for i in range(n))
        if ss_xx == 0:
            return 0.0, 0.0, 0.0
        beta = ss_xy / ss_xx
        alpha = mean_y - beta * mean_x
        y_pred = [alpha + beta * x for x in log_x]
        ss_res = sum((log_y[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((log_y[i] - mean_y) ** 2 for i in range(n))
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
        return math.exp(alpha), beta, r2

    def validates_scaling_law(self, data_sizes: List[int], recalls: List[float]) -> bool:
        _, beta, r2 = self.fit_power_law(data_sizes, recalls)
        return r2 >= self.min_r2 and beta > 0


def test_long_tail_rewriter():
    rewriter = LongTailQueryRewriter()
    q = ProductQuery("breast pump", "baby pump", is_long_tail=True)
    variants = rewriter.rewrite(q, n_variants=3)
    assert len(variants) >= 2
    assert any("pump" in v.lower() for v in variants)
    print(f"[PASS] long_tail_rewrite: {len(variants)} variants: {variants[:2]}")


def test_cross_domain_synthesis():
    synth = CrossDomainSynthesizer(transfer_noise=0.05)
    source = [SyntheticSample(f"baby pump {i}", f"sku_{i}", float(i % 2), domain="amazon") for i in range(20)]
    generated = synth.synthesize_from_source(source, "tiktok", n_samples=50)
    assert len(generated) == 50
    assert all(s.domain == "tiktok" for s in generated)
    assert all(0.0 <= s.label <= 1.0 for s in generated)
    print(f"[PASS] cross_domain: {len(generated)} synthetic samples, avg_label={sum(s.label for s in generated)/len(generated):.3f}")


def test_scaling_law_validation():
    validator = ScalingLawValidator(min_r_squared=0.8)
    sizes = [100, 500, 1000, 5000, 10000]
    recalls = [0.30, 0.42, 0.52, 0.68, 0.75]
    assert validator.validates_scaling_law(sizes, recalls)
    flat_recalls = [0.50, 0.51, 0.50, 0.51, 0.50]
    assert not validator.validates_scaling_law(sizes, flat_recalls)
    print(f"[PASS] scaling_law: power_law=True, flat=False")


def test_full_pipeline():
    random.seed(42)
    rewriter = LongTailQueryRewriter()
    synth = CrossDomainSynthesizer()
    queries = [ProductQuery("baby monitor", "monitor", is_long_tail=i % 2 == 0) for i in range(5)]
    all_samples = []
    for q in queries:
        variants = rewriter.rewrite(q, n_variants=2)
        source = [SyntheticSample(v, f"sku_{j}", random.random(), domain="amazon") for j, v in enumerate(variants)]
        generated = synth.synthesize_from_source(source, "tiktok_shop", n_samples=10)
        all_samples.extend(generated)
    assert len(all_samples) > 0
    assert all(s.source == "cross_domain_transfer" for s in all_samples)
    print(f"[PASS] full_pipeline: {len(all_samples)} total synthetic samples")


if __name__ == "__main__":
    test_long_tail_rewriter()
    test_cross_domain_synthesis()
    test_scaling_law_validation()
    test_full_pipeline()
    print("\n✅ All tests passed")
