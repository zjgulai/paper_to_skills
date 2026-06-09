"""
Document Intelligence Parsing
整合 dots.mocr + Qianfan-OCR + MinerU-Popo

论文来源:
  dots.mocr:    arXiv:2603.13032
  Qianfan-OCR:  arXiv:2603.13398
  MinerU-Popo:  arXiv:2605.24973
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class DocumentElementType(Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    TITLE = "title"


@dataclass
class ParsedTable:
    headers: List[List[str]]
    rows: List[List[str]]
    caption: str = ""
    page_start: int = 0
    page_end: int = 0

    def to_dicts(self) -> List[Dict[str, str]]:
        if not self.headers or not self.rows:
            return []
        flat = self.headers[-1] if len(self.headers) > 1 else self.headers[0]
        return [
            {flat[i]: row[i] for i in range(min(len(flat), len(row)))}
            for row in self.rows
        ]


@dataclass
class ParsedFigure:
    figure_type: str
    caption: str
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    certifications: List[str] = field(default_factory=list)


@dataclass
class ParsedDocument:
    title: str
    tables: List[ParsedTable]
    figures: List[ParsedFigure]
    text_blocks: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class LayoutAwareParser:
    MULTI_HEADER_PATTERNS = [
        r"(单价|价格|price).*(含税|不含税|tax|excl)",
        r"(数量|quantity|qty).*(最小|最大|min|max)",
        r"(尺寸|size|dimension).*(长|宽|高|length|width|height)",
    ]

    def infer_layout(self, raw_text: str) -> Dict[str, Any]:
        lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
        pipe_counts = [l.count("|") for l in lines if "|" in l]
        return {
            "estimated_columns": max(set(pipe_counts), key=pipe_counts.count) if pipe_counts else 1,
            "has_multi_row_header": any(
                re.search(p, " ".join(lines[:5]), re.I) for p in self.MULTI_HEADER_PATTERNS
            ),
            "has_merged_cells": bool(re.search(r'合计|总计|total|subtotal', raw_text, re.I)),
        }

    def parse_table(self, raw_text: str) -> ParsedTable:
        layout = self.infer_layout(raw_text)
        lines = [l.strip() for l in raw_text.split("\n") if l.strip() and "|" in l]
        headers, data_rows, header_done = [], [], False

        for line in lines:
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if not cells:
                continue
            if not header_done:
                headers.append(cells)
                if not layout["has_multi_row_header"] or len(headers) >= 2:
                    header_done = True
            else:
                data_rows.append(cells)

        return ParsedTable(headers=headers, rows=data_rows)


class CrossPageTableMerger:
    def __init__(self, min_col_overlap: float = 0.7):
        self.min_col_overlap = min_col_overlap

    def _is_truncated(self, t: ParsedTable, nxt: Optional[ParsedTable]) -> bool:
        if nxt is None or not t.rows or not nxt.rows:
            return False
        if any(re.search(r'合计|总计|total|subtotal', c, re.I)
               for c in t.rows[-1]):
            return False
        tc = len(t.headers[-1]) if t.headers else len(t.rows[0]) if t.rows else 0
        nc = len(nxt.headers[-1]) if nxt.headers else len(nxt.rows[0]) if nxt.rows else 0
        if tc == 0 or nc == 0:
            return False
        overlap = min(tc, nc) / max(tc, nc)
        nxt_no_header = not nxt.headers
        return overlap >= self.min_col_overlap and nxt_no_header

    def merge_tables(self, tables: List[ParsedTable]) -> List[ParsedTable]:
        if not tables:
            return []
        merged, i = [], 0
        while i < len(tables):
            cur = tables[i]
            while i + 1 < len(tables) and self._is_truncated(cur, tables[i + 1]):
                nxt = tables[i + 1]
                cur = ParsedTable(
                    headers=cur.headers,
                    rows=cur.rows + nxt.rows,
                    caption=cur.caption or nxt.caption,
                    page_start=cur.page_start,
                    page_end=nxt.page_end,
                )
                i += 1
            merged.append(cur)
            i += 1
        return merged


class MultimodalExtractor:
    CERTS = {"CE": r'\bCE\b', "FDA": r'\bFDA\b', "RoHS": r'\bRoHS\b',
             "REACH": r'\bREACH\b', "EN71": r'\bEN\s*71\b', "ASTM": r'\bASTM\b'}
    DIM_RE = re.compile(
        r'(\d+\.?\d*)\s*[×xX]\s*(\d+\.?\d*)\s*(?:[×xX]\s*(\d+\.?\d*))?\s*(cm|mm|inch|in|")',
        re.I
    )

    def extract_certifications(self, text: str) -> List[str]:
        return [c for c, p in self.CERTS.items() if re.search(p, text, re.I)]

    def extract_dimensions(self, text: str) -> Optional[Dict[str, Any]]:
        m = self.DIM_RE.search(text)
        if not m:
            return None
        unit = m.group(4).lower().replace('"', 'inch')
        vals = [float(v) for v in [m.group(1), m.group(2), m.group(3)] if v]
        result: Dict[str, Any] = {"unit": unit}
        if len(vals) >= 2:
            result["length"], result["width"] = vals[0], vals[1]
        if len(vals) == 3:
            result["height"] = vals[2]
        return result

    def extract_figure_data(self, figure_text: str, figure_type: str = "unknown") -> ParsedFigure:
        dims = self.extract_dimensions(figure_text)
        return ParsedFigure(
            figure_type=figure_type,
            caption=figure_text[:100],
            extracted_data={"dimensions": dims} if dims else {},
            certifications=self.extract_certifications(figure_text),
        )


class DocumentIntelligenceParser:
    def __init__(self):
        self.layout_parser = LayoutAwareParser()
        self.merger = CrossPageTableMerger()
        self.extractor = MultimodalExtractor()

    def parse_document(self, pages: List[Dict[str, str]]) -> ParsedDocument:
        tables, figures, texts = [], [], []
        title = ""
        for page_num, page in enumerate(pages):
            raw = page.get("text", "")
            ptype = page.get("type", "text")
            if page_num == 0 and raw:
                first = raw.split("\n")[0].strip()
                if len(first) < 100:
                    title = first
            if "|" in raw:
                t = self.layout_parser.parse_table(raw)
                if t.rows:
                    t.page_start = t.page_end = page_num
                    tables.append(t)
            if ptype == "figure" or re.search(r'认证|证书|certif|图纸|dimension', raw, re.I):
                figures.append(self.extractor.extract_figure_data(raw, ptype))
            else:
                clean = re.sub(r'\|[^|]+', '', raw).strip()
                if clean:
                    texts.append(clean)
        return ParsedDocument(
            title=title,
            tables=self.merger.merge_tables(tables),
            figures=figures,
            text_blocks=texts,
            metadata={"pages": len(pages)},
        )

    def extract_supplier_skus(self, doc: ParsedDocument) -> List[Dict[str, Any]]:
        sku_kw = {"sku", "item", "产品", "商品", "货号", "型号", "model", "code"}
        price_kw = {"price", "单价", "价格", "usd", "rmb", "cny", "¥", "$"}
        moq_kw = {"moq", "最小", "minimum", "起订", "min qty"}
        lead_kw = {"lead", "交期", "delivery", "days", "天", "week"}

        results = []
        all_certs = [c for fig in doc.figures for c in fig.certifications]

        for table in doc.tables:
            if not table.headers or not table.rows:
                continue
            flat = table.headers[-1]
            col_map: Dict[str, int] = {}
            for i, h in enumerate(flat):
                hl = h.lower()
                if any(k in hl for k in sku_kw):
                    col_map["sku"] = i
                elif any(k in hl for k in price_kw) and "price" not in col_map:
                    col_map["price"] = i
                elif any(k in hl for k in moq_kw):
                    col_map["moq"] = i
                elif any(k in hl for k in lead_kw):
                    col_map["lead_time"] = i
            if not col_map:
                continue
            for row in table.rows:
                entry: Dict[str, Any] = {
                    fn: row[ci] for fn, ci in col_map.items() if ci < len(row)
                }
                if entry:
                    if all_certs:
                        entry["certifications"] = list(set(all_certs))
                    results.append(entry)
        return results


def test_layout_aware_parser():
    parser = LayoutAwareParser()
    raw = "产品名称 | 单价(USD) | 单价(RMB)\n含税 | 不含税 | 含税\nSKU-001 | 0.85 | 0.80 | 5.90\nSKU-002 | 0.72 | 0.68 | 5.20"
    table = parser.parse_table(raw)
    assert len(table.headers) >= 1
    assert len(table.rows) >= 1
    print(f"[PASS] layout_parser: headers={len(table.headers)}, rows={len(table.rows)}")


def test_cross_page_merger():
    merger = CrossPageTableMerger()
    t1 = ParsedTable(headers=[["SKU", "Price", "MOQ"]], rows=[["A", "0.85", "500"], ["B", "0.72", "1000"]], page_end=1)
    t2 = ParsedTable(headers=[], rows=[["C", "0.65", "2000"], ["D", "0.58", "5000"]], page_start=2)
    merged = merger.merge_tables([t1, t2])
    assert len(merged) == 1
    assert len(merged[0].rows) == 4
    print(f"[PASS] cross_page_merge: {len(t1.rows)}+{len(t2.rows)} rows → {len(merged[0].rows)} merged")


def test_multimodal_extractor():
    extractor = MultimodalExtractor()
    text = "产品通过 CE, FDA, RoHS 认证，尺寸：12.5×8.0×5.0 cm"
    fig = extractor.extract_figure_data(text, "product_spec")
    assert "CE" in fig.certifications
    assert "FDA" in fig.certifications
    assert "RoHS" in fig.certifications
    assert fig.extracted_data.get("dimensions", {}).get("length") == 12.5
    print(f"[PASS] multimodal_extract: certs={fig.certifications}, dims={fig.extracted_data['dimensions']}")


def test_supplier_sku_extraction():
    parser = DocumentIntelligenceParser()
    pages = [
        {"text": "ABC 供应商 2026 年报价单\n产品型号 | 含税单价(USD) | 最小起订量 | 交期(天)\nBST-001 | 0.85 | 500 | 35\nBST-002 | 0.72 | 1000 | 28", "type": "text"},
        {"text": "产品通过 CE FDA RoHS 认证，尺寸 12×8×5 cm", "type": "figure"},
    ]
    doc = parser.parse_document(pages)
    skus = parser.extract_supplier_skus(doc)
    assert len(skus) >= 1
    assert "sku" in skus[0] or "price" in skus[0]
    print(f"[PASS] sku_extraction: {len(skus)} SKUs, sample={skus[0]}")


def test_cross_page_document():
    parser = DocumentIntelligenceParser()
    pages = [
        {"text": "供应商产能报告\n车间编号 | 月产能(万件) | 可用率\nA01 | 50 | 80%\nA02 | 35 | 90%", "type": "text"},
        {"text": "车间编号 | 月产能(万件) | 可用率\nA03 | 45 | 75%\nA04 | 60 | 85%", "type": "text"},
    ]
    doc = parser.parse_document(pages)
    assert len(doc.tables) >= 1
    total_rows = sum(len(t.rows) for t in doc.tables)
    print(f"[PASS] cross_page_doc: {len(doc.tables)} tables, {total_rows} total rows")


if __name__ == "__main__":
    test_layout_aware_parser()
    test_cross_page_merger()
    test_multimodal_extractor()
    test_supplier_sku_extraction()
    test_cross_page_document()
    print("\n✅ All tests passed")
