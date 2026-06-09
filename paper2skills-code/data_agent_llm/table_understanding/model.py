"""
Multimodal Table Understanding Agent — 表格理解：规格对比/认证矩阵/价格表
paper2skills-code: 09-DataAgent-LLM | 母婴出海跨境电商

纯 Python 标准库实现（无外部依赖）
Python 3.14 兼容
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union


# ──────────────────────────────────────────────
# 数据类：表格单元格
# ──────────────────────────────────────────────

@dataclass
class TableCell:
    """表格单元格"""
    row: int
    col: int
    value: Any
    cell_type: str = "data"


# ──────────────────────────────────────────────
# 表格类：序列化 + 查询操作
# ──────────────────────────────────────────────

class Table:
    """
    结构化矩形表格，支持 Markdown 序列化和行列级查询操作。

    内部存储：headers (list[str]) + rows (list[list[Any]])
    所有查询操作返回新的 Table 实例或计算结果，不修改原表格。
    """

    def __init__(self, headers: list[str], rows: list[list[Any]]) -> None:
        if rows and any(len(r) != len(headers) for r in rows):
            raise ValueError("所有行的列数必须与表头一致")
        self.headers = headers
        self.rows = [list(r) for r in rows]

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return len(self.headers)

    def _col_index(self, column: str) -> int:
        if column not in self.headers:
            raise KeyError(f"列 '{column}' 不存在，可用列: {self.headers}")
        return self.headers.index(column)

    def serialize_to_markdown(self) -> str:
        sep = "| " + " | ".join("---" for _ in self.headers) + " |"
        header_line = "| " + " | ".join(str(h) for h in self.headers) + " |"
        row_lines = [
            "| " + " | ".join(str(v) for v in row) + " |"
            for row in self.rows
        ]
        return "\n".join([header_line, sep] + row_lines)

    def get_column(self, column: str) -> list[Any]:
        idx = self._col_index(column)
        return [row[idx] for row in self.rows]

    def filter_rows(
        self,
        column: str,
        op: str,
        value: Any,
    ) -> "Table":
        """
        按条件过滤行，返回新 Table。

        支持操作符：eq, ne, gt, lt, gte, lte, contains
        """
        idx = self._col_index(column)
        ops: dict[str, Callable[[Any, Any], bool]] = {
            "eq": lambda a, b: str(a).strip() == str(b).strip(),
            "ne": lambda a, b: str(a).strip() != str(b).strip(),
            "gt": lambda a, b: float(a) > float(b),
            "lt": lambda a, b: float(a) < float(b),
            "gte": lambda a, b: float(a) >= float(b),
            "lte": lambda a, b: float(a) <= float(b),
            "contains": lambda a, b: str(b).strip() in str(a).strip(),
        }
        if op not in ops:
            raise ValueError(f"不支持的操作符: {op}，支持: {list(ops.keys())}")
        check = ops[op]
        filtered = [row for row in self.rows if check(row[idx], value)]
        return Table(headers=self.headers, rows=filtered)

    def aggregate(self, column: str, func: str) -> float:
        """
        对指定列执行聚合运算。

        支持：sum, avg, min, max, count
        """
        vals_raw = self.get_column(column)
        if func == "count":
            return float(len(vals_raw))

        try:
            vals = [float(v) for v in vals_raw]
        except (ValueError, TypeError) as exc:
            raise ValueError(f"列 '{column}' 含非数值，无法聚合: {exc}") from exc

        funcs: dict[str, Callable[[list[float]], float]] = {
            "sum": sum,
            "avg": lambda v: sum(v) / len(v) if v else 0.0,
            "min": min,
            "max": max,
        }
        if func not in funcs:
            raise ValueError(f"不支持的聚合函数: {func}，支持: {list(funcs.keys()) + ['count']}")
        return funcs[func](vals)

    def get_row_by_key(self, key_column: str, key_value: Any) -> Optional[list[Any]]:
        idx = self._col_index(key_column)
        for row in self.rows:
            if str(row[idx]).strip() == str(key_value).strip():
                return row
        return None

    def get_cell(self, key_column: str, key_value: Any, target_column: str) -> Any:
        row = self.get_row_by_key(key_column, key_value)
        if row is None:
            raise KeyError(f"未找到 {key_column}={key_value} 的行")
        col_idx = self._col_index(target_column)
        return row[col_idx]


# ──────────────────────────────────────────────
# Table QA Agent
# ──────────────────────────────────────────────

@dataclass
class QueryResult:
    """查询结果"""
    query_type: str
    answer: Any
    rows_matched: int = 0
    explanation: str = ""


class TableQAAgent:
    """
    表格问答 Agent，支持过滤/比较/聚合三类查询。

    设计原则：将查询类型显式化，避免 LLM 黑盒推理错误；
    所有计算操作在结构化层完成，LLM 仅负责自然语言理解和结果渲染。
    """

    def __init__(self, table: Table) -> None:
        self._table = table

    def execute_query(
        self,
        query_type: str,
        column: Optional[str] = None,
        op: Optional[str] = None,
        value: Any = None,
        conditions: Optional[list[dict]] = None,
    ) -> QueryResult:
        """
        执行结构化查询。

        Args:
            query_type: "filter" | "aggregate" | "compare"
            column: 目标列名
            op: 操作符（filter 用）
            value: 比较值（filter 用）
            conditions: 多条件过滤列表 [{"column": ..., "op": ..., "value": ...}]
        """
        if query_type == "filter":
            return self._filter(column, op, value, conditions)
        if query_type == "aggregate":
            return self._aggregate(column, op or "avg")
        if query_type == "compare":
            return self._compare(conditions or [])
        raise ValueError(f"不支持的查询类型: {query_type}")

    def _filter(
        self,
        column: Optional[str],
        op: Optional[str],
        value: Any,
        conditions: Optional[list[dict]],
    ) -> QueryResult:
        result_table = self._table

        if conditions:
            for cond in conditions:
                result_table = result_table.filter_rows(
                    column=cond["column"],
                    op=cond["op"],
                    value=cond["value"],
                )
        elif column and op:
            result_table = result_table.filter_rows(column=column, op=op, value=value)

        first_col = result_table.headers[0]
        matched = [str(row[0]) for row in result_table.rows]
        return QueryResult(
            query_type="filter",
            answer=matched,
            rows_matched=len(matched),
            explanation=f"过滤结果: {len(matched)} 行满足条件 → {matched}",
        )

    def _aggregate(self, column: Optional[str], func: str) -> QueryResult:
        if not column:
            raise ValueError("聚合查询需要指定 column")
        result = self._table.aggregate(column, func)
        return QueryResult(
            query_type="aggregate",
            answer=result,
            explanation=f"{func.upper()}({column}) = {result:.4f}",
        )

    def _compare(self, conditions: list[dict]) -> QueryResult:
        results = {}
        for cond in conditions:
            row_key = cond.get("row_key")
            col_target = cond.get("column")
            key_col = cond.get("key_column", self._table.headers[0])
            if row_key and col_target:
                results[row_key] = self._table.get_cell(key_col, row_key, col_target)
        return QueryResult(
            query_type="compare",
            answer=results,
            explanation=f"比较结果: {results}",
        )

    def compare_rows(self, row_a: str, row_b: str, column: str) -> float:
        key_col = self._table.headers[0]
        val_a = float(self._table.get_cell(key_col, row_a, column))
        val_b = float(self._table.get_cell(key_col, row_b, column))
        return val_a - val_b

    def aggregate(self, column: str, func: str = "avg") -> float:
        return self._table.aggregate(column, func)


# ──────────────────────────────────────────────
# 样本数据：婴儿奶粉规格对比表 (5×8)
# ──────────────────────────────────────────────

def build_formula_table() -> Table:
    """构建 5 款婴儿奶粉规格对比表（5行×8列）"""
    headers = ["品牌", "阶段", "含HMO", "价格($/lb)", "有机认证", "铁含量(mg)", "DHA", "产地"]
    rows = [
        ["Brand A", "Stage 2", "是", 42.5, "是", 1.8, "是", "美国"],
        ["Brand B", "Stage 2", "否", 38.0, "否", 2.1, "是", "荷兰"],
        ["Brand C", "Stage 2", "是", 55.0, "是", 1.5, "否", "爱尔兰"],
        ["Brand D", "Stage 2", "是", 47.0, "否", 1.9, "是", "美国"],
        ["Brand E", "Stage 2", "否", 33.0, "是", 2.3, "否", "德国"],
    ]
    return Table(headers=headers, rows=rows)


def build_certification_matrix() -> Table:
    """构建工厂认证矩阵表（4行×5认证）"""
    headers = ["工厂", "FDA注册", "EU_IFP", "UK_UKCA", "ISO22000", "FSSC22000"]
    rows = [
        ["Factory 1", "✓", "✓", "✓", "✓", "✗"],
        ["Factory 2", "✓", "✓", "✗", "✓", "✓"],
        ["Factory 3", "✓", "✗", "✓", "✗", "✓"],
        ["Factory 4", "✓", "✓", "✓", "✓", "✓"],
    ]
    return Table(headers=headers, rows=rows)


# ──────────────────────────────────────────────
# 测试：3 类查询验证
# ──────────────────────────────────────────────

def _run_tests() -> None:
    print("=" * 60)
    print("Multimodal Table Understanding Agent — 表格查询测试")
    print("=" * 60)

    formula_table = build_formula_table()
    cert_table = build_certification_matrix()

    print(f"\n[表格结构] 奶粉规格表: {formula_table.n_rows} 行 × {formula_table.n_cols} 列")
    print(formula_table.serialize_to_markdown())

    # 查询 1：过滤查询 — 含HMO 且 价格 < 50
    print(f"\n[查询 1] 过滤: 含HMO=是 且 价格<$50/lb")
    agent = TableQAAgent(formula_table)
    result = agent.execute_query(
        query_type="filter",
        conditions=[
            {"column": "含HMO", "op": "eq", "value": "是"},
            {"column": "价格($/lb)", "op": "lt", "value": 50.0},
        ],
    )
    print(f"  结果: {result.answer}")
    print(f"  说明: {result.explanation}")
    assert "Brand A" in result.answer, "Brand A 应在结果中（含HMO, $42.5）"
    assert "Brand D" in result.answer, "Brand D 应在结果中（含HMO, $47.0）"
    assert "Brand C" not in result.answer, "Brand C 应被过滤（价格 $55 > $50）"
    print(f"  ✓ 过滤查询验证通过")

    # 查询 2：聚合查询 — 所有品牌平均价格
    print(f"\n[查询 2] 聚合: 平均价格")
    avg_price = agent.aggregate("价格($/lb)", func="avg")
    expected_avg = (42.5 + 38.0 + 55.0 + 47.0 + 33.0) / 5
    print(f"  平均价格: ${avg_price:.2f} (预期: ${expected_avg:.2f})")
    assert abs(avg_price - expected_avg) < 0.01, f"平均价格计算错误: {avg_price} != {expected_avg}"
    max_price = agent.aggregate("价格($/lb)", func="max")
    assert max_price == 55.0, f"最高价应为 55.0，实际: {max_price}"
    print(f"  最高价格: ${max_price:.2f}")
    print(f"  ✓ 聚合查询验证通过")

    # 查询 3：比较查询 — Brand A vs Brand D 价格差
    print(f"\n[查询 3] 比较: Brand A vs Brand D 价格差")
    price_diff = agent.compare_rows("Brand A", "Brand D", column="价格($/lb)")
    print(f"  Brand A - Brand D 价格差: ${price_diff:.1f}")
    assert abs(price_diff - (42.5 - 47.0)) < 0.01, f"价格差计算错误: {price_diff}"
    print(f"  ✓ 比较查询验证通过")

    # 认证矩阵查询
    print(f"\n[查询 4] 认证矩阵: FDA + EU_IFP + UK_UKCA 三证齐全的工厂")
    print(cert_table.serialize_to_markdown())
    cert_agent = TableQAAgent(cert_table)
    cert_result = cert_agent.execute_query(
        query_type="filter",
        conditions=[
            {"column": "FDA注册", "op": "eq", "value": "✓"},
            {"column": "EU_IFP", "op": "eq", "value": "✓"},
            {"column": "UK_UKCA", "op": "eq", "value": "✓"},
        ],
    )
    print(f"  三证齐全工厂: {cert_result.answer}")
    assert "Factory 1" in cert_result.answer, "Factory 1 应三证齐全"
    assert "Factory 4" in cert_result.answer, "Factory 4 应三证齐全"
    assert "Factory 2" not in cert_result.answer, "Factory 2 缺 UK_UKCA"
    assert "Factory 3" not in cert_result.answer, "Factory 3 缺 EU_IFP"
    print(f"  ✓ 认证矩阵查询验证通过")

    print("\n" + "=" * 60)
    print("[✓] 所有场景验证通过 — Multimodal Table Understanding Agent")
    print("=" * 60)


if __name__ == "__main__":
    _run_tests()
