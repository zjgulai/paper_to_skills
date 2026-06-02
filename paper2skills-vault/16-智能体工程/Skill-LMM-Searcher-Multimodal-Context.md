---
title: LMM-Searcher — 长链多模态 Agent：UID 占位符按需加载图片
doc_type: knowledge
module: 16-智能体工程
topic: lmm-searcher-multimodal-context
status: stable
created: 2026-06-01
updated: 2026-06-01
owner: self
source: human+ai
---

# Skill Card: LMM-Searcher — 长链多模态 Agent 上下文管理（UID 占位符 + 按需加载）

**来源**：LMM-Searcher: Long-horizon Agentic Multimodal Search | arXiv:2604.12890 | 2026年4月
**代码**：[github.com/RUCAIBox/LMM-Searcher](https://github.com/RUCAIBox/LMM-Searcher)

---

## ① 算法原理

### 核心思想

**LMM-Searcher** 解决长链多模态 Agent 的**上下文爆炸**问题：在 100 轮搜索会话中，若每张图片直接嵌入为 base64（约 1,000-3,000 tokens），50 张图片就会占用 50,000-150,000 tokens，远超实用预算。

**三层架构**：

1. **UID 占位符机制**：每张图片在注册时分配唯一 ID（如 `<IMG_UID_0042>`），在 Agent 上下文中仅保留 UID + 文字描述（~30 tokens），图片数据存入外部 `ImageRegistry`（文件系统或内存）。

2. **fetch-image 主动感知工具**：Agent 通过工具调用 `fetch_image(uid)` 按需拉取图片到当前 context window 中。只有在需要"视觉细节比较"时才触发 fetch，节省 70%+ token。典型决策点：文字描述无法区分时（颜色差异、标签细节、图表数值）。

3. **蒸馏训练数据构建**：研究团队构建 12K 条高质量 long-horizon 搜索轨迹（含 UID 占位 → fetch 决策的完整序列），蒸馏微调 Qwen3-VL-30A3B，模型习得"何时应 fetch 图片"的主动决策能力。在 MM-BrowseComp 基准上达到 SOTA，支持 100 轮超长会话。

**token 预算分析**：100 轮 × 平均 5 张图/轮 = 500 张图。UID 模式仅占 500 × 30 = 15K tokens；若全嵌入需 500 × 2,000 = 100 万 tokens，差距 67 倍。

---

## ② 母婴出海应用案例

### 场景一：选品 Agent 多图竞品分析

**业务问题**：选品 Agent 一次搜索 10 个竞品，每个竞品有 6-8 张产品图（主图、详情图、包装图），Agent 需跨图对比外观、认证标志、卖点差异。

**传统做法**：全部图片嵌入 context，10 竞品 × 7 图 × 2,000 tokens = **14 万 tokens**，每次查询成本约 ¥2.1。

**UID 占位模式**：
- 注册阶段：70 张图存入 ImageRegistry，context 仅含 UID 占位串（70 × 30 = 2,100 tokens）
- Agent 决策：仅在需要"细节对比"时 fetch（通常 5-8 张），按需加载 5 × 2,000 = 10,000 tokens
- **token 减少约 70%**，成本降至 ¥0.63/次
- **每月节省**（100 次/天 × 30 天）：约 **¥180-300**

**数据要求**：产品图 URL 或本地路径；Agent 需支持工具调用（tool_use / function_calling）

**预期产出**：结构化竞品报告（外观差异 + 认证对比 + 卖点矩阵），可直接供选品决策

### 场景二：合规 Agent 多认证文档审查

**业务问题**：一款婴儿玩具上架德国需同时满足 CE、EN71、REACH 等 20+ 认证要求，每个认证对应 PDF 图片页（测试报告截图、证书扫描件）。

**UID 占位模式**：
- 20 份认证文档 × 平均 3 页图片 = 60 张图，全部注册为 UID
- 合规 Agent 按审查序列逐项触发 fetch_image：先 fetch CE 证书封面页验证有效期，再 fetch EN71 测试项页核查参数
- 每次审查仅加载当前核查项图片（1-3 张），context 始终可控
- **支持持续 100 轮审查会话**，不因上下文爆炸而重置

**业务价值**：一次完整合规审查从"需要多轮对话重置" → "单次会话完成"，减少审查时间 50%

---

## ③ 代码模板

> 📄 代码位置：`paper2skills-code/llm_agent_engineering/lmm_searcher_context/model.py`

```python
"""
LMM-Searcher: UID 占位符 + 按需加载图片的长链多模态 Agent 上下文管理
参考: arXiv:2604.12890 — LMM-Searcher: Long-horizon Agentic Multimodal Search
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────
# 1. ImageRegistry：图片外部存储与 UID 管理
# ─────────────────────────────────────────────

class ImageRegistry:
    """
    外部图片注册中心。图片以 base64 存储在内存 dict 中（生产环境可替换为文件系统/对象存储）。
    register_image() → uid；fetch_image(uid) → base64 字符串
    """

    def __init__(self):
        self._store: dict[str, str] = {}         # uid → base64
        self._meta: dict[str, dict] = {}          # uid → metadata
        self._counter: int = 0

    def register_image(
        self,
        source: str | bytes,
        description: str = "",
        metadata: Optional[dict] = None,
    ) -> str:
        """
        注册图片，返回 UID。
        source: 本地文件路径、URL字符串，或原始 bytes
        """
        uid = f"IMG_UID_{self._counter:04d}"
        self._counter += 1

        if isinstance(source, bytes):
            b64 = base64.b64encode(source).decode()
        elif isinstance(source, str) and os.path.isfile(source):
            with open(source, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
        elif isinstance(source, str) and source.startswith("http"):
            # 生产环境替换为真实 HTTP 下载；此处用 mock 数据
            mock_bytes = f"[MOCK_IMAGE_DATA:{source}]".encode()
            b64 = base64.b64encode(mock_bytes).decode()
        else:
            b64 = base64.b64encode(str(source).encode()).decode()

        self._store[uid] = b64
        self._meta[uid] = {
            "description": description,
            "source": str(source)[:100],
            "registered_at": time.time(),
            **(metadata or {}),
        }
        return uid

    def fetch_image(self, uid: str) -> Optional[str]:
        """按 UID 加载图片 base64，不存在返回 None"""
        return self._store.get(uid)

    def get_meta(self, uid: str) -> Optional[dict]:
        return self._meta.get(uid)

    @property
    def size(self) -> int:
        return len(self._store)


# ─────────────────────────────────────────────
# 2. UIDPlaceholder：UID 占位符数据类
# ─────────────────────────────────────────────

@dataclass
class UIDPlaceholder:
    """
    在 Agent 上下文中代替真实图片的轻量占位符。
    token 开销：约 30 tokens（uid + description）
    """
    uid: str
    description: str
    metadata: dict = field(default_factory=dict)

    def to_context_str(self) -> str:
        """生成写入 Agent 上下文的占位文本"""
        return f"<{self.uid}>[{self.description}]"

    def to_dict(self) -> dict:
        return {
            "uid": self.uid,
            "description": self.description,
            "metadata": self.metadata,
        }


# ─────────────────────────────────────────────
# 3. LMMSearcherContext：长链上下文管理器
# ─────────────────────────────────────────────

class LMMSearcherContext:
    """
    管理长链多模态 Agent 的上下文。
    - 图片以 UID 占位符存储，不直接嵌入 base64
    - 按需通过 fetch_image_for_context() 加载图片
    - 追踪 token 消耗统计（UID 模式 vs 全量嵌入）
    """

    TOKENS_PER_IMAGE_FULL = 2000    # 全量嵌入一张图的估算 token 数
    TOKENS_PER_UID = 30             # UID 占位符估算 token 数

    def __init__(self, registry: ImageRegistry):
        self.registry = registry
        self._placeholders: dict[str, UIDPlaceholder] = {}  # uid → placeholder
        self._messages: list[dict] = []                       # Agent 对话历史
        self._fetch_log: list[str] = []                       # 已 fetch 的 uid 列表

    def add_image(self, source: str | bytes, description: str, metadata: Optional[dict] = None) -> UIDPlaceholder:
        """注册图片并返回 UID 占位符（写入 context 前调用）"""
        uid = self.registry.register_image(source, description, metadata)
        placeholder = UIDPlaceholder(uid=uid, description=description, metadata=metadata or {})
        self._placeholders[uid] = placeholder
        return placeholder

    def fetch_image_for_context(self, uid: str) -> Optional[str]:
        """
        按需加载图片 base64（模拟 fetch-image 工具）。
        返回 base64 字符串，供 VLM 调用时直接嵌入。
        """
        if uid not in self._placeholders:
            return None
        b64 = self.registry.fetch_image(uid)
        if b64:
            self._fetch_log.append(uid)
        return b64

    def get_context_with_placeholder(self) -> str:
        """
        返回含 UID 占位符的文本上下文（不含图片 base64）。
        这是传递给 LLM 的主 context 字符串。
        """
        lines = []
        for uid, ph in self._placeholders.items():
            lines.append(ph.to_context_str())
        return "\n".join(lines)

    def add_message(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content, "timestamp": time.time()})

    def token_stats(self) -> dict:
        """对比 UID 模式 vs 全量嵌入的 token 消耗"""
        n_images = len(self._placeholders)
        n_fetched = len(self._fetch_log)
        uid_tokens = n_images * self.TOKENS_PER_UID + n_fetched * self.TOKENS_PER_IMAGE_FULL
        full_tokens = n_images * self.TOKENS_PER_IMAGE_FULL
        saved = full_tokens - uid_tokens
        pct = saved / full_tokens * 100 if full_tokens else 0
        return {
            "total_images": n_images,
            "fetched_images": n_fetched,
            "uid_mode_tokens": uid_tokens,
            "full_embed_tokens": full_tokens,
            "tokens_saved": saved,
            "savings_pct": round(pct, 1),
        }


# ─────────────────────────────────────────────
# 4. FetchImageTool：模拟 fetch-image 工具调用
# ─────────────────────────────────────────────

class FetchImageTool:
    """
    模拟 LMM-Searcher 中的 fetch-image 工具。
    Agent 通过调用此工具主动拉取图片到 context。
    """

    def __init__(self, context: LMMSearcherContext):
        self.context = context
        self.call_count = 0

    @property
    def tool_definition(self) -> dict:
        return {
            "name": "fetch_image",
            "description": "Fetch image by UID when visual details are needed for comparison",
            "parameters": {
                "type": "object",
                "properties": {
                    "uid": {"type": "string", "description": "Image UID from context placeholder"},
                    "reason": {"type": "string", "description": "Why this image is needed now"},
                },
                "required": ["uid"],
            },
        }

    def __call__(self, uid: str, reason: str = "") -> dict:
        """执行 fetch-image 工具调用"""
        self.call_count += 1
        b64 = self.context.fetch_image_for_context(uid)
        meta = self.context.registry.get_meta(uid)

        if b64 is None:
            return {"success": False, "error": f"UID {uid} not found", "uid": uid}

        return {
            "success": True,
            "uid": uid,
            "base64_preview": b64[:50] + "...",  # 实际使用完整 base64
            "description": meta.get("description", "") if meta else "",
            "reason": reason,
            "call_index": self.call_count,
        }


# ─────────────────────────────────────────────
# 5. 测试：50 张图片选品场景对比
# ─────────────────────────────────────────────

def test_selection_agent_scenario():
    """
    测试场景：50 张竞品图片搜索
    对比：全量嵌入 vs UID 占位模式的 token 消耗
    """
    print("=" * 60)
    print("LMM-Searcher UID 占位符 vs 全量嵌入 对比测试")
    print("=" * 60)

    registry = ImageRegistry()
    ctx = LMMSearcherContext(registry)
    fetch_tool = FetchImageTool(ctx)

    # 模拟注册 50 张竞品图片（10 个竞品 × 5 图/竞品）
    competitors = [f"竞品_{chr(65+i)}" for i in range(10)]
    image_types = ["主图", "详情图1", "详情图2", "包装图", "认证图"]

    print(f"\n[Step 1] 注册 50 张竞品图片（仅存 UID 占位）...")
    placeholders: list[UIDPlaceholder] = []
    for comp in competitors:
        for img_type in image_types:
            # 模拟图片 URL
            mock_url = f"https://cdn.example.com/{comp}/{img_type}.jpg"
            description = f"{comp} {img_type}"
            ph = ctx.add_image(mock_url, description, {"competitor": comp, "type": img_type})
            placeholders.append(ph)

    print(f"  注册完成：{registry.size} 张图片")
    print(f"  Context 占位符预览（前3个）:")
    for ph in placeholders[:3]:
        print(f"    {ph.to_context_str()}")

    # 模拟 Agent 决策：只 fetch 8 张（需要视觉细节的）
    print(f"\n[Step 2] Agent 按需 fetch（仅 8 张需要视觉对比的图片）...")
    fetch_targets = [
        (placeholders[0].uid, "需要对比竞品A和竞品B的主图外观差异"),
        (placeholders[5].uid, "需要确认竞品B认证图上的CE标志有效期"),
        (placeholders[10].uid, "需要查看竞品C的包装图设计风格"),
        (placeholders[15].uid, "竞品D详情图中标注的材质与描述不符"),
        (placeholders[20].uid, "竞品E主图与搜索词匹配度需目视确认"),
        (placeholders[25].uid, "竞品F的认证图证书编号需要核实"),
        (placeholders[30].uid, "竞品G包装图与竞品H差异对比"),
        (placeholders[35].uid, "竞品H详情图展示的尺寸标注需核查"),
    ]

    for uid, reason in fetch_targets:
        result = fetch_tool(uid, reason)
        status = "✓" if result["success"] else "✗"
        print(f"  {status} fetch {uid}: {reason[:30]}...")

    # 输出统计
    print(f"\n[Step 3] Token 消耗对比统计:")
    stats = ctx.token_stats()
    print(f"  总图片数: {stats['total_images']} 张")
    print(f"  实际 fetch: {stats['fetched_images']} 张")
    print(f"  UID 模式 token: {stats['uid_mode_tokens']:,}")
    print(f"  全量嵌入 token: {stats['full_embed_tokens']:,}")
    print(f"  节省 token: {stats['tokens_saved']:,} ({stats['savings_pct']}%)")

    cost_per_1k = 0.003  # Claude 3.5 Haiku 估算
    uid_cost = stats["uid_mode_tokens"] / 1000 * cost_per_1k
    full_cost = stats["full_embed_tokens"] / 1000 * cost_per_1k
    print(f"\n  成本估算（$0.003/1K tokens）:")
    print(f"  UID 模式: ${uid_cost:.4f}")
    print(f"  全量嵌入: ${full_cost:.4f}")
    print(f"  每次搜索节省: ${full_cost - uid_cost:.4f}")
    monthly_savings = (full_cost - uid_cost) * 3000  # 100次/天×30天
    print(f"  月节省（100次/天）: ${monthly_savings:.2f}")

    # 验证
    assert stats["total_images"] == 50, "应注册 50 张图片"
    assert stats["fetched_images"] == 8, "应 fetch 8 张图片"
    assert stats["savings_pct"] > 60, "UID 模式节省比例应 > 60%"
    assert fetch_tool.call_count == 8, "工具调用次数应为 8"

    print(f"\n✅ 所有断言通过！")
    return stats


if __name__ == "__main__":
    test_selection_agent_scenario()
```

---

## ④ 技能关联

**前置**：
- [[Skill-Context-Compression]] — Agent 历史压缩基础机制
- [[Skill-Active-Context-Pruning]] — 主动裁剪上下文的互补方案
- [[Skill-VLM-Ecommerce-Adaptation]] — 电商场景 VLM 调优基础

**延伸**：
- [[Skill-Agentic-Memory-Management]] — 长期记忆与外部存储进阶
- [[Skill-Memory-as-Action]] — 把"存储操作"建模为 Agent 动作

**可组合**：
- [[Skill-Shopping-Companion-Agent]] — 选品 Agent 直接集成 ImageRegistry
- [[Skill-Listing-Quality-Scoring]] — 图片 UID 供质量评分 Agent 按需加载
- [[Skill-Tool-Call-Decision-Framework]] — 决策何时触发 fetch-image 工具调用

---

## ⑤ 商业价值评估

| 维度 | 评估 |
|------|------|
| **ROI 量化** | token 减少 70%，100 次/天选品 Agent 月省 $150-300 |
| **核心价值** | 支持 100 轮超长会话，Agent 连续工作不中断重置 |
| **适用规模** | 中大型跨境品牌（日搜索 50+ 次），小品牌 ROI 有限 |
| **实施难度** | ⭐⭐☆☆☆（ImageRegistry 独立于 LLM 框架，易集成） |
| **优先级** | ⭐⭐⭐⭐☆（多图场景必备，单图场景可跳过） |
| **风险** | Agent 未触发 fetch 时可能遗漏视觉细节（需训练决策能力） |
