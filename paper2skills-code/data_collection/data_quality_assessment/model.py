"""
E-commerce Data Quality Assessment
整合 MESReduce + MMPCBench + macrOData

论文来源:
  MESReduce:   arXiv:2603.08612
  MMPCBench:   arXiv:2601.19750
  macrOData:   arXiv:2602.09329
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ProductRecord:
    sku: str
    title: str = ""
    price: float = 0.0
    category: str = ""
    image_url: str = ""
    description: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    quality_issues: List[str] = field(default_factory=list)
    quality_score: float = 1.0


class MESErrorDetector:
    """MESReduce 风格：Maximal Error Score 量化标注错误对查询的影响"""

    def compute_mes(self, record: ProductRecord, required_fields: List[str]) -> float:
        missing = [f for f in required_fields if not getattr(record, f, None) and f not in record.attributes]
        if not missing:
            return 0.0
        return len(missing) / len(required_fields)

    def detect_errors(self, record: ProductRecord) -> List[str]:
        errors = []
        if not record.title or len(record.title) < 5:
            errors.append("missing_or_short_title")
        if record.price <= 0:
            errors.append("invalid_price")
        if record.price > 10000:
            errors.append("price_out_of_range")
        if not record.image_url:
            errors.append("missing_image")
        if not record.category:
            errors.append("missing_category")
        if not record.description or len(record.description) < 20:
            errors.append("missing_or_short_description")
        return errors

    def score_record(self, record: ProductRecord) -> Tuple[float, List[str]]:
        errors = self.detect_errors(record)
        required = ["title", "price", "image_url", "category", "description"]
        mes = self.compute_mes(record, required)
        issue_penalty = len(errors) * 0.15
        quality = max(0.0, 1.0 - mes - issue_penalty)
        return quality, errors


class MissingModalityDetector:
    """MMPCBench 风格：检测商品 catalog 中的缺失模态"""

    MODALITIES = ["title", "image_url", "description", "price", "category"]
    COMPLETENESS_THRESHOLDS = {"title": 10, "description": 50}

    def check_modalities(self, record: ProductRecord) -> Dict[str, bool]:
        result = {}
        for modality in self.MODALITIES:
            val = getattr(record, modality, None) or record.attributes.get(modality)
            if modality in self.COMPLETENESS_THRESHOLDS:
                result[modality] = bool(val) and len(str(val)) >= self.COMPLETENESS_THRESHOLDS[modality]
            elif modality == "price":
                result[modality] = isinstance(val, (int, float)) and val > 0
            else:
                result[modality] = bool(val)
        return result

    def completeness_score(self, record: ProductRecord) -> float:
        checks = self.check_modalities(record)
        return sum(checks.values()) / len(checks)

    def suggest_fixes(self, record: ProductRecord) -> List[Dict[str, str]]:
        checks = self.check_modalities(record)
        suggestions = []
        for modality, ok in checks.items():
            if not ok:
                suggestions.append({
                    "field": modality,
                    "issue": f"missing_or_incomplete_{modality}",
                    "action": f"fill_{modality}_from_supplier_data",
                })
        return suggestions


class OutlierDetector:
    """macrOData 风格：表格异常值检测"""

    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination

    def detect_price_outliers(self, records: List[ProductRecord]) -> List[str]:
        prices = [r.price for r in records if r.price > 0]
        if len(prices) < 3:
            return []
        sorted_p = sorted(prices)
        n = len(sorted_p)
        q1 = sorted_p[n // 4]
        q3 = sorted_p[(3 * n) // 4]
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return [r.sku for r in records if r.price > 0 and (r.price < lower or r.price > upper)]

    def detect_category_anomalies(self, records: List[ProductRecord]) -> List[str]:
        from collections import Counter
        cats = Counter(r.category for r in records if r.category)
        if not cats:
            return []
        total = sum(cats.values())
        rare_threshold = max(1, total * self.contamination)
        rare_cats = {c for c, cnt in cats.items() if cnt <= rare_threshold}
        return [r.sku for r in records if r.category in rare_cats]


class DataQualityPipeline:
    def __init__(self):
        self.error_detector = MESErrorDetector()
        self.modality_detector = MissingModalityDetector()
        self.outlier_detector = OutlierDetector()

    def assess(self, records: List[ProductRecord]) -> Dict[str, Any]:
        scores, all_issues = [], []
        price_outliers = self.outlier_detector.detect_price_outliers(records)
        cat_anomalies = self.outlier_detector.detect_category_anomalies(records)

        for record in records:
            quality, errors = self.error_detector.score_record(record)
            completeness = self.modality_detector.completeness_score(record)
            combined = 0.6 * quality + 0.4 * completeness

            if record.sku in price_outliers:
                errors.append("price_outlier")
                combined = max(0.0, combined - 0.2)

            record.quality_score = combined
            record.quality_issues = errors
            scores.append(combined)
            if errors:
                all_issues.extend(errors)

        from collections import Counter
        return {
            "total": len(records),
            "avg_quality": sum(scores) / max(len(scores), 1),
            "high_quality": sum(1 for s in scores if s >= 0.8),
            "needs_attention": sum(1 for s in scores if s < 0.5),
            "common_issues": Counter(all_issues).most_common(5),
            "price_outliers": len(price_outliers),
        }


def test_error_detection():
    detector = MESErrorDetector()
    bad = ProductRecord("BAD-001", title="Hi", price=-1.0)
    quality, errors = detector.score_record(bad)
    assert quality < 0.5
    assert "invalid_price" in errors
    assert "missing_or_short_title" in errors
    print(f"[PASS] error_detect: quality={quality:.3f}, errors={errors}")


def test_modality_completeness():
    detector = MissingModalityDetector()
    complete = ProductRecord("G-001", title="Baby Monitor Pro Max Smart Camera",
                              price=89.99, category="electronics",
                              image_url="http://img.com/g001.jpg",
                              description="HD baby monitor with night vision and two-way audio for parents")
    incomplete = ProductRecord("B-001", title="Item", price=0.0)
    assert detector.completeness_score(complete) > detector.completeness_score(incomplete)
    fixes = detector.suggest_fixes(incomplete)
    assert len(fixes) > 0
    print(f"[PASS] modality: complete={detector.completeness_score(complete):.2f}, fixes_needed={len(fixes)}")


def test_outlier_detection():
    detector = OutlierDetector()
    records = [ProductRecord(f"sku{i}", price=float(90 + i)) for i in range(8)]
    records.append(ProductRecord("outlier", price=9999.0))
    outliers = detector.detect_price_outliers(records)
    assert "outlier" in outliers
    print(f"[PASS] outlier_detect: found {len(outliers)} outliers: {outliers}")


def test_full_pipeline():
    pipeline = DataQualityPipeline()
    records = [
        ProductRecord("A1", "Momcozy M9 Electric Breast Pump Double Portable Wearable", 89.99, "breast_pump",
                      "http://img.com/a1.jpg", "Revolutionary wearable breast pump with strong suction motor"),
        ProductRecord("A2", "", 0.0),
        ProductRecord("A3", "Baby Monitor", 9999.0, "electronics", "http://img.com/a3.jpg",
                      "Smart baby monitor with HD camera and alerts for parents"),
    ]
    report = pipeline.assess(records)
    assert report["total"] == 3
    assert report["high_quality"] >= 1
    assert report["needs_attention"] >= 1
    print(f"[PASS] full_pipeline: avg={report['avg_quality']:.3f}, high={report['high_quality']}, issues={report['common_issues'][:2]}")


if __name__ == "__main__":
    test_error_detection()
    test_modality_completeness()
    test_outlier_detection()
    test_full_pipeline()
    print("\n✅ All tests passed")
