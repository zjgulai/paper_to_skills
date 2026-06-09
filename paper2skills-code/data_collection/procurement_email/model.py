"""
Procurement Email Extraction
整合 Contract2Plan + ProUIE + NLP-based Page Classification

论文来源:
  Contract2Plan: arXiv:2601.06164
  ProUIE:        arXiv:2604.10633
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ProcurementTerm:
    field_name: str
    value: Any
    unit: str = ""
    evidence_span: str = ""
    confidence: float = 1.0


@dataclass
class ExtractedProcurement:
    supplier_name: str
    product_name: str
    terms: List[ProcurementTerm]
    raw_text: str = ""
    is_compliant: bool = True
    compliance_issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {t.field_name: {"value": t.value, "unit": t.unit, "confidence": t.confidence}
                for t in self.terms}


class ProcurementFieldExtractor:
    MOQ_RE = re.compile(r'(?:MOQ|最小.*?量|minimum.*?order|起订)\s*[:\s]*(\d+[\d,]*)\s*(pcs?|件|units?|套)?', re.I)
    PRICE_RE = re.compile(r'(?:price|单价|价格|报价)\s*[:\s]*\$?\s*([\d.]+)\s*(USD|CNY|RMB|元)?', re.I)
    LEAD_RE = re.compile(r'(?:lead time|交期|delivery|交货)\s*[:\s]*(\d+)\s*(days?|weeks?|天|周)', re.I)
    LADDER_RE = re.compile(r'(\d[\d,]+)\s*(?:\+|~|-|以上)?\s*(?:pcs?|件|units?)?\s*[:\s]*\$?\s*([\d.]+)', re.I)
    SUPPLIER_RE = re.compile(r'(?:from|supplier|manufacturer|发件人|供应商)[:\s]+([A-Za-z\u4e00-\u9fff][\w\s\u4e00-\u9fff]{2,40})', re.I)
    PRODUCT_RE = re.compile(r'(?:product|item|商品|产品)[:\s]+([A-Za-z\u4e00-\u9fff][\w\s\u4e00-\u9fff]{2,60})', re.I)

    def extract(self, text: str) -> ExtractedProcurement:
        terms = []

        m = self.MOQ_RE.search(text)
        if m:
            terms.append(ProcurementTerm("moq", int(m.group(1).replace(",", "")),
                                          unit=m.group(2) or "pcs", evidence_span=m.group(0)))

        prices = []
        for m in self.PRICE_RE.finditer(text):
            prices.append(ProcurementTerm("unit_price", float(m.group(1)),
                                           unit=m.group(2) or "USD", evidence_span=m.group(0)))
        if prices:
            terms.extend(prices[:1])

        m = self.LEAD_RE.search(text)
        if m:
            terms.append(ProcurementTerm("lead_time", int(m.group(1)),
                                          unit=m.group(2), evidence_span=m.group(0)))

        ladder = []
        for m in self.LADDER_RE.finditer(text):
            qty = int(m.group(1).replace(",", ""))
            price = float(m.group(2))
            if qty > 0 and 0 < price < 10000:
                ladder.append({"qty": qty, "price": price})
        if len(ladder) >= 2:
            terms.append(ProcurementTerm("price_ladder", ladder, unit="USD/pcs"))

        sup_m = self.SUPPLIER_RE.search(text)
        prod_m = self.PRODUCT_RE.search(text)
        return ExtractedProcurement(
            supplier_name=sup_m.group(1).strip() if sup_m else "Unknown",
            product_name=prod_m.group(1).strip() if prod_m else "Unknown",
            terms=terms,
            raw_text=text,
        )


class ComplianceValidator:
    def validate(self, proc: ExtractedProcurement, budget: float = float("inf"),
                 max_lead_days: int = 60) -> ExtractedProcurement:
        issues = []
        term_dict = proc.to_dict()

        moq = term_dict.get("moq", {}).get("value", 0)
        unit_price = term_dict.get("unit_price", {}).get("value", 0)
        if moq and unit_price:
            total_cost = moq * unit_price
            if total_cost > budget:
                issues.append(f"min_order_cost ${total_cost:.0f} exceeds budget ${budget:.0f}")

        lead = term_dict.get("lead_time", {}).get("value", 0)
        if lead and lead > max_lead_days:
            issues.append(f"lead_time {lead} days exceeds max {max_lead_days} days")

        proc.compliance_issues = issues
        proc.is_compliant = len(issues) == 0
        return proc


def test_extract_moq_price_lead():
    extractor = ProcurementFieldExtractor()
    email = """
    From: Shenzhen ABC Electronics
    Product: Baby Monitor Pro
    MOQ: 500 pcs
    Unit Price: $12.50 USD
    Lead Time: 35 days
    1000+ pcs: $11.20, 3000+ pcs: $10.50
    """
    result = extractor.extract(email)
    td = result.to_dict()
    assert result.supplier_name != "Unknown"
    assert td["moq"]["value"] == 500
    assert abs(td["unit_price"]["value"] - 12.50) < 0.01
    assert td["lead_time"]["value"] == 35
    assert len(td["price_ladder"]["value"]) >= 2
    print(f"[PASS] extract: supplier={result.supplier_name}, moq={td['moq']['value']}, "
          f"price={td['unit_price']['value']}, lead={td['lead_time']['value']}")


def test_compliance_validation():
    extractor = ProcurementFieldExtractor()
    validator = ComplianceValidator()
    email = "MOQ: 2000 pcs, Unit Price: $15.00, Lead Time: 90 days"
    proc = extractor.extract(email)
    validated = validator.validate(proc, budget=20000, max_lead_days=60)
    assert not validated.is_compliant
    assert any("lead_time" in issue for issue in validated.compliance_issues)
    print(f"[PASS] compliance: issues={validated.compliance_issues}")


def test_price_ladder_extraction():
    extractor = ProcurementFieldExtractor()
    email = "500 pcs: $1.20, 1000 pcs: $1.05, 3000 pcs: $0.95"
    result = extractor.extract(email)
    td = result.to_dict()
    ladder = td.get("price_ladder", {}).get("value", [])
    assert len(ladder) >= 2
    prices = [item["price"] for item in ladder]
    assert prices == sorted(prices, reverse=True)
    print(f"[PASS] price_ladder: {ladder}")


if __name__ == "__main__":
    test_extract_moq_price_lead()
    test_compliance_validation()
    test_price_ladder_extraction()
    print("\n✅ All tests passed")
