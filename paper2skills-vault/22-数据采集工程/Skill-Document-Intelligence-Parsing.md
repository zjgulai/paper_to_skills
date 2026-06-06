---
title: Document Intelligence Parsing — LLM 驱动的文档智能解析：图文统一 OCR、跨页表格恢复、布局感知推理
doc_type: knowledge
module: 22-数据采集工程
topic: document-intelligence-parsing
status: stable
created: 2026-06-05
updated: 2026-06-05
owner: self
source: human+ai
---

# Skill Card: Document Intelligence Parsing — 文档智能解析

> **图谱定位**：Layer 1 基础层｜解锁 `Skill-Multi-SKU-Procurement-Budget-Allocation`、`Skill-Supplier-Capacity-Planning`、`Skill-Dynamic-Lot-Sizing-MOQ` 的数据源前置

---

## ① 算法原理

### 核心思想

供应商发来的报价单是 PDF，工厂产能表是 Excel 截图，海关 HS 编码文件是扫描件——这些"已有但不可用"的数据是母婴跨境电商最大的数据孤岛。传统 OCR（Tesseract）只能识别文字，无法理解**表格结构、跨页截断、图文混排**。

2026 年三篇论文从互补角度解决文档解析的三大难题：

| 论文 | 解决的核心问题 | 关键机制 |
|------|-------------|---------|
| **dots.mocr** (2603.13032) | 图形/图表/SVG 无法结构化输出 | Multimodal OCR：文本+图形统一为结构化 JSON |
| **Qianfan-OCR** (2603.13398) | 布局理解与文字识别割裂 | Layout-as-Thought：`<think>` token 驱动布局推理 |
| **MinerU-Popo** (2605.24973) | 跨页表格截断丢失上下文 | 专用后处理：跨页表格拼接恢复 |

### dots.mocr：图文统一 OCR

**核心创新**：传统 OCR 只输出纯文本，遇到图表/流程图/SVG 就跳过或输出乱码。dots.mocr 将文档中**所有内容**（包括图形）统一转化为结构化输出。

**架构**：3B 参数 Multimodal LLM，输入文档页面图片，输出格式化 Markdown/JSON：

```
输入：PDF 页面截图（含文字 + 表格 + 图表 + SVG）
                ↓
Vision Encoder（ViT-based）→ 提取视觉特征
                ↓
Text Decoder（LLM）→ 生成结构化输出
                ↓
输出：
  {
    "text_blocks": [...],
    "tables": [{"headers": [...], "rows": [[...], ...]}],
    "figures": [{"caption": "...", "type": "bar_chart", "data": {...}}]
  }
```

**关键结果**：
- olmOCR-Bench：**83.9 分**（SOTA，开源模型第一）
- 比 GPT-4o 在文档解析专项任务上得分更高
- 完全开源，本地可部署

### Qianfan-OCR：Layout-as-Thought

**核心创新**：在 LLM 生成输出之前，先让模型**"思考"文档布局**（类似 Chain-of-Thought），避免把多栏文本读成一行、把跨列表头读错。

**Layout-as-Thought 机制**：

```
普通 OCR：
  输入图片 → 直接输出文字（不理解"这是两列并排"）

Layout-as-Thought：
  输入图片
    ↓
  <think>
    观察到：左侧为商品名称列，右侧为价格列
    表头跨越两行，第一行为"单价"，第二行为"含税"
    第 3-7 行为数据行，第 8 行为合计行
  </think>
    ↓
  输出结构化表格（正确理解列对齐和合并单元格）
```

**关键结果**：
- OmniDocBench v1.5 端到端模型排名**第一（93.12）**
- 表格 TEDS（Tree Edit Distance Score）：**91.02**（越接近 100 越好）
- 4B 参数，推理速度快

### MinerU-Popo：跨页表格恢复

**问题**：一张供应商价格表跨越 3 页 PDF，每页截断的表格是独立的，标准 OCR 输出 3 个破碎表格而非 1 个完整表格。

**MinerU-Popo 后处理流程**：

```
Step 1: 截断检测
  检测每页表格末尾是否为"截断行"（无完整语义的行）
  检测下一页表格开头是否为"续行"（无新表头）

Step 2: 表格拼接
  匹配跨页同一表格（列数相同 + 语义连续）
  合并为完整表格，恢复行索引

Step 3: 标题层级修复
  使用 4B 模型预测标题层级（H1/H2/H3）
  修复因跨页导致的标题层级紊乱

Step 4: 输出统一格式
  兼容 Markdown / JSON / Excel 输出
  支持对接 RAG pipeline（降低 embedding 的 token 消耗 70%）
```

**关键结果**：
- 标题层级 TEDS 提升 **≥20%**
- RAG 延迟降低 **70%**（完整表格的 chunk 更语义完整，检索更精准）
- 支持 MinerU / Marker / Docling 等主流 OCR 引擎作为前端

---

## ② 母婴出海应用案例

### 场景一：供应商报价单批量解析（WF-A 采购流水线）

**业务背景**：每季度收到 15-20 家供应商的 PDF 报价单，每份 5-30 页，包含 SKU 表格（商品名/规格/MOQ/阶梯价格/交期）。目前人工录入需要 3-4 天，错误率约 8%。

**三工具协作流程**：

```
供应商 PDF（含跨页价格表 + 图文混排产品说明）
    ↓
Qianfan-OCR（Layout-as-Thought）
    识别合并单元格表头："单价（USD）"跨 3 列
    正确区分"含税价"和"不含税价"两个子列
    ↓
MinerU-Popo（跨页表格拼接）
    SKU-001 到 SKU-023：跨第 4、5 页的价格表完整合并
    标题层级：第一页"主产品线" > 第三页"配件清单"
    ↓
dots.mocr（图形提取）
    产品尺寸图 → {"length": 12.5, "width": 8.0, "unit": "cm"}
    认证标志图片 → {"certifications": ["CE", "FDA", "RoHS"]}
    ↓
结构化输出 JSON：
  [{"sku": "BST-001", "name": "婴儿安抚奶嘴", "moq": 500,
    "price_usd": {"1000+": 0.85, "3000+": 0.72, "5000+": 0.65},
    "lead_time_days": 35, "certifications": ["CE", "FDA"]}]
    ↓
自动输入 Skill-Multi-SKU-Procurement-Budget-Allocation

效果：录入时间 3-4天 → 2小时（自动化率 85%），错误率 8% → <1%
```

### 场景二：进口清关文件合规预检（WF-D 选品扫描）

**业务背景**：新品进入欧盟市场前需要检查 CE 声明、REACH 合规报告（通常为 PDF 扫描件），判断是否满足合规要求。

**应用效果**：
- dots.mocr 从合规 PDF 中提取检测项目表格
- Qianfan-OCR 正确识别化学品限制值表格（多层表头：物质名 / 测试方法 / 限制值 / 检测值 / 结论）
- 自动对比 REACH 法规阈值 → 生成合规预检报告
- 时间：人工 2h/份 → 自动化 5min/份

---

## ③ 代码模板

代码位置：`paper2skills-code/data_collection/document_intelligence/model.py`

```python
"""
Document Intelligence Parsing
整合 dots.mocr (图文统一) + Qianfan-OCR (布局推理) + MinerU-Popo (跨页表格)

论文来源:
  dots.mocr:    arXiv:2603.13032
  Qianfan-OCR:  arXiv:2603.13398
  MinerU-Popo:  arXiv:2605.24973
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


class DocumentElementType(Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    TITLE = "title"
    LIST = "list"


@dataclass
class TableCell:
    value: str
    row_span: int = 1
    col_span: int = 1
    is_header: bool = False


@dataclass
class ParsedTable:
    headers: List[List[str]]
    rows: List[List[str]]
    caption: str = ""
    page_start: int = 0
    page_end: int = 0
    is_truncated: bool = False

    def to_dicts(self) -> List[Dict[str, str]]:
        if not self.headers or not self.rows:
            return []
        flat_headers = self.headers[-1] if len(self.headers) > 1 else self.headers[0]
        return [
            {flat_headers[i]: row[i] for i in range(min(len(flat_headers), len(row)))}
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
    """
    Qianfan-OCR 风格：Layout-as-Thought 布局感知解析
    在输出结构化内容前，先推理文档布局
    """

    MULTI_HEADER_PATTERNS = [
        r"(单价|价格|price).*(含税|不含税|tax|excl)",
        r"(数量|quantity|qty).*(最小|最大|min|max)",
        r"(尺寸|size|dimension).*(长|宽|高|length|width|height)",
    ]

    def __init__(self):
        self.layout_cache: Dict[str, str] = {}

    def infer_layout(self, raw_text: str) -> Dict[str, Any]:
        """
        Step 1: 推理文档布局（模拟 <think> 过程）
        返回布局元数据：列数、是否多行表头、是否跨列
        """
        lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
        layout = {
            "estimated_columns": self._estimate_columns(lines),
            "has_multi_row_header": self._detect_multi_row_header(lines),
            "has_merged_cells": self._detect_merged_cells(raw_text),
            "table_count": raw_text.count("|") // 10,
        }
        return layout

    def _estimate_columns(self, lines: List[str]) -> int:
        pipe_counts = [l.count("|") for l in lines if "|" in l]
        if not pipe_counts:
            return 1
        return max(set(pipe_counts), key=pipe_counts.count)

    def _detect_multi_row_header(self, lines: List[str]) -> bool:
        for pattern in self.MULTI_HEADER_PATTERNS:
            combined = " ".join(lines[:5])
            if re.search(pattern, combined, re.I):
                return True
        return False

    def _detect_merged_cells(self, text: str) -> bool:
        return bool(re.search(r'colspan|rowspan|合计|subtotal|total', text, re.I))

    def parse_table(self, raw_text: str) -> ParsedTable:
        """
        Layout-aware 表格解析：先推理布局，再解析结构
        """
        layout = self.infer_layout(raw_text)
        lines = [l.strip() for l in raw_text.split("\n") if l.strip() and "|" in l]

        headers, data_rows = [], []
        header_done = False

        for line in lines:
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if not cells:
                continue
            if not header_done:
                if layout["has_multi_row_header"] and len(headers) < 2:
                    headers.append(cells)
                else:
                    header_done = True
                    headers.append(cells)
                    header_done = True
            else:
                if all(c.replace(".", "").replace(",", "").replace("-", "").isdigit()
                       or c in ("", "N/A", "-") for c in cells):
                    data_rows.append(cells)
                elif not any(c.isdigit() for c in "".join(cells)):
                    if len(headers) < 2:
                        headers.append(cells)
                    else:
                        data_rows.append(cells)
                else:
                    data_rows.append(cells)

        if not headers:
            return ParsedTable(headers=[], rows=data_rows)
        return ParsedTable(headers=headers, rows=data_rows)


class CrossPageTableMerger:
    """
    MinerU-Popo 风格：跨页表格截断检测与拼接恢复
    """

    def __init__(self, min_col_overlap: float = 0.7):
        self.min_col_overlap = min_col_overlap

    def detect_truncation(self, table: ParsedTable, next_table: Optional[ParsedTable]) -> bool:
        """
        判断 table 是否被截断（应与 next_table 合并）
        条件：① 当前表末尾无合计行 ② 下一表无表头 ③ 列数相似
        """
        if next_table is None:
            return False
        if not table.rows or not next_table.rows:
            return False

        last_row = table.rows[-1] if table.rows else []
        has_summary = any(
            re.search(r'合计|总计|total|subtotal|sum', str(c), re.I)
            for c in last_row
        )
        if has_summary:
            return False

        t_cols = len(table.headers[-1]) if table.headers else 0
        n_cols = len(next_table.headers[-1]) if next_table.headers else 0
        if t_cols == 0 or n_cols == 0:
            return False

        col_overlap = min(t_cols, n_cols) / max(t_cols, n_cols)
        next_has_no_header = (
            not next_table.headers or
            len(next_table.headers) == 0 or
            all(
                any(char.isdigit() for char in cell)
                for cell in (next_table.headers[0] if next_table.headers else [])
            )
        )
        return col_overlap >= self.min_col_overlap and next_has_no_header

    def merge_tables(self, tables: List[ParsedTable]) -> List[ParsedTable]:
        """
        遍历所有表格，合并跨页截断的表格
        """
        if not tables:
            return []

        merged = []
        i = 0
        while i < len(tables):
            current = tables[i]
            while i + 1 < len(tables) and self.detect_truncation(current, tables[i + 1]):
                next_t = tables[i + 1]
                current = ParsedTable(
                    headers=current.headers,
                    rows=current.rows + next_t.rows,
                    caption=current.caption or next_t.caption,
                    page_start=current.page_start,
                    page_end=next_t.page_end,
                    is_truncated=False,
                )
                i += 1
            merged.append(current)
            i += 1
        return merged


class MultimodalExtractor:
    """
    dots.mocr 风格：从图形中提取结构化信息
    （认证标志、尺寸图、图表数据）
    """

    CERTIFICATION_PATTERNS = {
        "CE": r'\bCE\b',
        "FDA": r'\bFDA\b',
        "RoHS": r'\bRoHS\b',
        "REACH": r'\bREACH\b',
        "EN71": r'\bEN\s*71\b',
        "ASTM": r'\bASTM\b',
        "CPSC": r'\bCPSC\b',
    }

    DIMENSION_PATTERN = re.compile(
        r'(\d+\.?\d*)\s*[×xX]\s*(\d+\.?\d*)\s*(?:[×xX]\s*(\d+\.?\d*))?\s*(cm|mm|inch|in|")',
        re.I
    )

    def extract_certifications(self, text: str) -> List[str]:
        found = []
        for cert, pattern in self.CERTIFICATION_PATTERNS.items():
            if re.search(pattern, text, re.I):
                found.append(cert)
        return found

    def extract_dimensions(self, text: str) -> Optional[Dict[str, Any]]:
        m = self.DIMENSION_PATTERN.search(text)
        if not m:
            return None
        unit = m.group(4).lower().replace('"', 'inch')
        dims = {"unit": unit}
        values = [float(v) for v in [m.group(1), m.group(2), m.group(3)] if v]
        if len(values) >= 2:
            dims["length"] = values[0]
            dims["width"] = values[1]
        if len(values) == 3:
            dims["height"] = values[2]
        return dims

    def extract_figure_data(self, figure_text: str, figure_type: str = "unknown") -> ParsedFigure:
        certs = self.extract_certifications(figure_text)
        dims = self.extract_dimensions(figure_text)
        extracted = {}
        if dims:
            extracted["dimensions"] = dims
        return ParsedFigure(
            figure_type=figure_type,
            caption=figure_text[:100],
            extracted_data=extracted,
            certifications=certs,
        )


class DocumentIntelligenceParser:
    """
    集成三层文档解析：Layout-Aware + CrossPage Merge + Multimodal
    """

    def __init__(self):
        self.layout_parser = LayoutAwareParser()
        self.cross_page_merger = CrossPageTableMerger()
        self.multimodal_extractor = MultimodalExtractor()

    def parse_document(self, pages: List[Dict[str, str]]) -> ParsedDocument:
        """
        输入：每页文档的文本内容列表
        输出：结构化 ParsedDocument
        """
        all_tables: List[ParsedTable] = []
        all_figures: List[ParsedFigure] = []
        all_text: List[str] = []
        title = ""

        for page_num, page in enumerate(pages):
            raw = page.get("text", "")
            page_type = page.get("type", "text")

            if page_num == 0 and raw:
                first_line = raw.split("\n")[0].strip()
                if len(first_line) < 100:
                    title = first_line

            if "|" in raw or "\t" in raw:
                table = self.layout_parser.parse_table(raw)
                if table.rows:
                    table.page_start = page_num
                    table.page_end = page_num
                    all_tables.append(table)

            if page_type == "figure" or "图" in raw or "figure" in raw.lower():
                figure = self.multimodal_extractor.extract_figure_data(raw, page_type)
                all_figures.append(figure)
            else:
                text_content = re.sub(r'\|[^|]+', '', raw).strip()
                if text_content:
                    all_text.append(text_content)

        merged_tables = self.cross_page_merger.merge_tables(all_tables)

        return ParsedDocument(
            title=title,
            tables=merged_tables,
            figures=all_figures,
            text_blocks=all_text,
            metadata={"total_pages": len(pages), "table_count": len(merged_tables)},
        )

    def extract_supplier_sku_data(self, document: ParsedDocument) -> List[Dict[str, Any]]:
        """
        专用于供应商报价单 SKU 数据提取
        识别 SKU/价格/MOQ/交期 等关键字段
        """
        sku_keywords = {"sku", "item", "产品", "商品", "货号", "型号", "model"}
        price_keywords = {"price", "单价", "价格", "usd", "rmb", "cny"}
        moq_keywords = {"moq", "最小", "minimum", "起订"}
        lead_keywords = {"lead", "交期", "delivery", "days", "天"}

        results = []
        for table in document.tables:
            if not table.headers or not table.rows:
                continue
            flat_headers = table.headers[-1]
            col_map = {}
            for i, h in enumerate(flat_headers):
                h_lower = h.lower()
                if any(k in h_lower for k in sku_keywords):
                    col_map["sku"] = i
                elif any(k in h_lower for k in price_keywords):
                    col_map["price"] = i
                elif any(k in h_lower for k in moq_keywords):
                    col_map["moq"] = i
                elif any(k in h_lower for k in lead_keywords):
                    col_map["lead_time"] = i

            if not col_map:
                continue

            for row in table.rows:
                entry = {}
                for field_name, col_idx in col_map.items():
                    if col_idx < len(row):
                        entry[field_name] = row[col_idx]
                if entry:
                    certs_from_doc = []
                    for fig in document.figures:
                        certs_from_doc.extend(fig.certifications)
                    if certs_from_doc:
                        entry["certifications"] = list(set(certs_from_doc))
                    results.append(entry)

        return results
```

---

## ④ 技能关联

### 前置技能
- 无（Layer 1 基础层，本 Skill 是数据源提供者）

### 延伸技能（本 Skill 输出数据驱动下游）
- [[Skill-Procurement-Email-Extraction]]：邮件提取 ← 文档解析的并行数据源
- [[Skill-Multi-SKU-Procurement-Budget-Allocation]]：解析结果直接输入
- [[Skill-Supplier-Capacity-Planning]]：产能数据来源
- [[Skill-Dynamic-Lot-Sizing-MOQ]]：MOQ 数据来源

### 可组合技能
- [[Skill-MAS-Dynamic-KG-Collaboration]]：解析结果写入动态 KG
- [[Skill-Cross-Org-Agent-Protocol]]：跨组织文档交换协议
- [[Skill-Ecommerce-Data-Quality-Assessment]]：解析后做质量检验

---

## ⑤ 商业价值评估

| 维度 | 评估 |
|------|------|
| **ROI 预估** | 供应商报价录入：3-4天/季度 → 2h（节省 ~24h/季度 × 人力成本）；错误率 8% → <1%（避免因价格录入错误导致的采购损失，单次错误影响 $5,000-$50,000）|
| **实施难度** | ⭐⭐☆☆☆（dots.mocr/Qianfan-OCR 开源可本地部署；MinerU-Popo 提供 PyPI 包；无需 API 费用）|
| **优先级评分** | ⭐⭐⭐⭐⭐（解锁供应链 3 个核心 Skill 的数据源，是 WF-A 智能补货的数据管道前置）|
| **评估依据** | Qianfan-OCR OmniDocBench 第一(93.12)；dots.mocr olmOCR-Bench SOTA(83.9)；MinerU-Popo RAG 延迟 -70%；三者均开源，生产可验证 |

---

## 论文来源

| 论文 | arXiv | 年份 | 开源 |
|------|-------|------|------|
| dots.mocr: Multimodal OCR Parse Anything | [2603.13032](https://arxiv.org/abs/2603.13032) | 2026-03 | ✅ |
| Qianfan-OCR: Layout-as-Thought Document Intelligence | [2603.13398](https://arxiv.org/abs/2603.13398) | 2026-03 | ✅ |
| MinerU-Popo: Universal Post-Processing for Structured Parsing | [2605.24973](https://arxiv.org/abs/2605.24973) | 2026-05 | ✅ |

---
## ⑥ Skill Relations
**前置技能（Prerequisite）**
- [[Skill-LLM-Focused-Web-Crawling]]

**延伸技能（Extends）**
- [[Skill-Procurement-Email-Extraction]]

**可组合技能（Combinable）**
- [[Skill-Ecommerce-Data-Quality-Assessment]]
- [[Skill-Multi-SKU-Procurement-Budget-Allocation]]

