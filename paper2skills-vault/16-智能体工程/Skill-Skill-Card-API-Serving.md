---
title: Skill Card API Serving — 将 Skill 代码模板包装为参数化 REST 微服务
doc_type: knowledge
module: 16-智能体工程
topic: skill-card-api-serving
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill Card: Skill Card API Serving

> **领域**：Agent 工程化 × 微服务封装 | **类型**: 工程基础
> **桥梁**: 16-智能体工程 ↔ 22-数据采集工程 | **2026年**

---

## ① 算法原理

### 核心思想

当前 Skill 卡片的代码模板只能在本地 Jupyter 或脚本中运行，Agent 无法通过网络调用。**Skill Card API Serving** 的核心是：**自动解析 Skill.md 的代码块 + frontmatter**，通过 FastAPI + Pydantic 将其注册为可参数化调用的 REST endpoint，实现「文档即服务」。

关键设计决策：
1. **Schema 自动推导**：从代码块的函数签名（`def run(param_a: str, param_b: int)`）自动生成 Pydantic 输入模型
2. **沙箱执行**：每次 API 调用在独立 Python 环境中 exec 代码模板，隔离副作用
3. **版本不变性**：每个 Skill 版本对应独立 endpoint path（`/skills/v1/demand-forecasting`），路由不覆盖

### 数学直觉

**endpoint 注册映射**：

$$\text{endpoint}_{s} = \text{``/skills/v1/''} + \text{topic}(s)$$

**请求生命周期**（延迟拆解）：

$$T_{\text{total}} = T_{\text{parse}} + T_{\text{validate}} + T_{\text{exec}} + T_{\text{serialize}}$$

目标：$T_{\text{total}} < 500\text{ms}$（代码执行为主要瓶颈，用进程池隔离）

### 关键假设

- Skill.md 的代码块中存在 `def run(...)` 入口函数
- 代码块无副作用（不写文件、不修改全局状态）
- 生产部署时使用 gunicorn/uvicorn 多进程

---

## ② 母婴出海应用案例

**场景 A：供应链 Agent 远程调用需求预测 Skill**

- **业务问题**：供应链 Agent 运行在 LLM 云端，无法直接 import 本地 Python 包；需要调用需求预测 Skill 获取 30 天预测值
- **数据要求**：Skill 已部署为微服务，Agent 只需发送 JSON（`{"asin": "B08XX", "history": [...]}`）
- **预期产出**：HTTP 200 + `{"forecast": [102, 98, 115, ...], "confidence_interval": [...]}`
- **业务价值**：Agent 与 Skill 解耦，可独立迭代；支持 A/B 测试不同版本 Skill；预测调用延迟 < 300ms，年化节省人工整合成本约 **22 万元**

**场景 B：多客户 SaaS 模式 Skill 按调用计费**

- **业务问题**：团队有 5 个品类运营团队，每个团队需要不同的 Skill 组合，希望按使用量计费而不是每人维护一套 Python 环境
- **数据要求**：统一 Skill 微服务平台，各团队通过 API Key 调用
- **预期产出**：调用日志 + 用量统计 + 异常告警；各团队按调用次数付费
- **业务价值**：5 个团队共用一套 Skill 服务，减少环境维护成本约 **15 万元/年**；新增 Skill 后全团队自动可用

---

## ③ 代码模板

```python
"""
Skill Card API Serving
FastAPI 自动将 Skill.md 代码块注册为 REST endpoint
依赖：无（mock 实现，生产环境需要 fastapi, uvicorn, pydantic）
"""

import re
import ast
import json
import time
import inspect
import hashlib
from typing import Any
from dataclasses import dataclass, field


# ─── Skill 解析器 ─────────────────────────────────────────────────────────────

@dataclass
class SkillEndpoint:
    skill_id: str
    topic: str
    version: str
    path: str
    code: str
    signature: dict[str, str]  # param_name -> type_annotation
    description: str


def parse_skill_md(md_content: str, skill_id: str) -> SkillEndpoint | None:
    """从 Skill.md 解析代码块和元数据"""
    # 提取 frontmatter
    fm_match = re.match(r'^---\n(.*?)\n---', md_content, re.DOTALL)
    if not fm_match:
        return None

    fm_text = fm_match.group(1)
    topic = re.search(r'topic:\s*(.+)', fm_text)
    topic = topic.group(1).strip() if topic else skill_id.lower()

    # 提取第一个 python 代码块中的 run() 函数
    _fence = "\x60\x60\x60"
    code_blocks = re.findall(_fence + r'python\n(.*?)' + _fence, md_content, re.DOTALL)
    run_code = None
    run_sig: dict[str, str] = {}

    for block in code_blocks:
        if 'def run(' in block:
            run_code = block
            # 解析 run() 函数签名
            try:
                tree = ast.parse(block)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == 'run':
                        for arg in node.args.args:
                            ann = ''
                            if arg.annotation:
                                ann = ast.unparse(arg.annotation)
                            run_sig[arg.arg] = ann
            except SyntaxError:
                pass
            break

    if not run_code:
        return None

    return SkillEndpoint(
        skill_id=skill_id,
        topic=topic,
        version="v1",
        path=f"/skills/v1/{topic}",
        code=run_code,
        signature=run_sig,
        description=f"Auto-generated endpoint for {skill_id}",
    )


# ─── 沙箱执行器 ───────────────────────────────────────────────────────────────

class SkillSandbox:
    """安全执行 Skill 代码块（mock 进程隔离，生产用 subprocess 或 RestrictedPython）"""

    def execute(self, endpoint: SkillEndpoint, params: dict[str, Any],
                timeout_s: float = 10.0) -> dict[str, Any]:
        t0 = time.perf_counter()
        namespace: dict[str, Any] = {}
        try:
            exec(endpoint.code, namespace)  # noqa: S102
            run_fn = namespace.get('run')
            if not run_fn or not callable(run_fn):
                return {"error": "run() function not found in Skill code", "status": 500}

            result = run_fn(**params)
            elapsed = (time.perf_counter() - t0) * 1000

            return {
                "status": 200,
                "skill_id": endpoint.skill_id,
                "path": endpoint.path,
                "result": result,
                "elapsed_ms": round(elapsed, 2),
            }
        except TypeError as e:
            return {"error": f"参数类型错误: {e}", "status": 422}
        except Exception as e:  # noqa: BLE001
            return {"error": str(e), "status": 500}


# ─── Mock FastAPI 路由注册 ────────────────────────────────────────────────────

class MockFastAPIApp:
    """模拟 FastAPI App 行为（不依赖真实 fastapi 包）"""

    def __init__(self):
        self.routes: dict[str, SkillEndpoint] = {}
        self.sandbox = SkillSandbox()
        self.call_log: list[dict] = []

    def register(self, endpoint: SkillEndpoint) -> None:
        self.routes[endpoint.path] = endpoint
        print(f"[注册] {endpoint.path} | params={list(endpoint.signature.keys())}")

    def call(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if path not in self.routes:
            return {"error": f"404 Not Found: {path}", "status": 404}

        endpoint = self.routes[path]
        response = self.sandbox.execute(endpoint, payload)
        self.call_log.append({
            "path": path,
            "payload": payload,
            "status": response.get("status"),
            "ts": time.time(),
        })
        return response


# ─── 测试用 Skill.md 片段 ─────────────────────────────────────────────────────

_FENCE = "\x60" * 3
MOCK_SKILL_MD = f"""---
title: 简单库存预测 Skill
doc_type: knowledge
module: 04-供应链
topic: simple-inventory-forecast
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase1
---

## ③ 代码模板

{_FENCE}python
def run(asin: str, history: list, horizon: int = 7):
    \"\"\"简单移动平均需求预测（测试用）\"\"\"
    if not history:
        return {{"error": "history empty"}}
    window = history[-min(7, len(history)):]
    avg = sum(window) / len(window)
    forecast = [round(avg * (1 + 0.02 * i), 1) for i in range(horizon)]
    return {{
        "asin": asin,
        "forecast": forecast,
        "method": "moving_average",
        "horizon": horizon,
    }}
{_FENCE}
"""


# ─── 测试用例 ──────────────────────────────────────────────────────────────────

def test_skill_card_api_serving():
    # 1. 解析 Skill.md
    endpoint = parse_skill_md(MOCK_SKILL_MD, "Skill-Simple-Inventory-Forecast")
    assert endpoint is not None, "Skill.md 解析失败"
    assert endpoint.path == "/skills/v1/simple-inventory-forecast"
    assert "asin" in endpoint.signature
    print(f"[✓] 解析成功: {endpoint.path}, params={endpoint.signature}")

    # 2. 注册到 Mock App
    app = MockFastAPIApp()
    app.register(endpoint)
    assert "/skills/v1/simple-inventory-forecast" in app.routes

    # 3. 正常调用
    resp = app.call(
        "/skills/v1/simple-inventory-forecast",
        {"asin": "B08XXXX", "history": [100, 110, 95, 120, 105], "horizon": 5},
    )
    assert resp["status"] == 200, f"调用失败: {resp}"
    assert "forecast" in resp["result"]
    assert len(resp["result"]["forecast"]) == 5
    print(f"[✓] 正常调用: forecast={resp['result']['forecast']}, "
          f"elapsed={resp['elapsed_ms']}ms")

    # 4. 404 路由测试
    resp404 = app.call("/skills/v1/nonexistent", {})
    assert resp404["status"] == 404
    print(f"[✓] 404 测试通过: {resp404['error']}")

    # 5. 调用日志
    assert len(app.call_log) == 2
    print(f"[✓] 调用日志: {len(app.call_log)} 条记录")

    print("\n[✓] Skill Card API Serving 测试通过")


if __name__ == "__main__":
    test_skill_card_api_serving()
```

---

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Agent-Skill-Runtime-Orchestrator]]（Orchestrator 需要调用 API 形态的 Skill）
- **延伸（extends）**：[[Skill-Agent-SLO-Manager]]（对 API 服务做 SLO 监控：延迟/可用性/错误率）
- **可组合（combinable）**：[[Skill-Agentic-Workflow-Compilation]]（将高频调用路径编译进 LLM 权重，减少 API 调用开销）
- **可组合（combinable）**：[[Skill-Data-Quality-Monitor-Alert]]（对 Skill API 输入数据做质量校验后再执行）

---

## ⑤ 商业价值评估

- **ROI 预估**：
  - 现状：每个 Agent 团队各维护一套 Python 环境，5 个团队 × 2 人 × 0.5 FTE 维护成本 = **50 万元/年**
  - 引入后：统一 Skill 微服务，维护成本降至 1 人 × 0.3 FTE ≈ **6 万元/年**
  - 年化节省：**约 44 万元**；另 Skill 迭代发布时间从 1 天 → 30 分钟（热更新）
  - 新业务接入新 Skill 的集成时间：从 2 天 → 2 小时
- **实施难度**：⭐⭐☆☆☆（FastAPI 标准开发，无特殊算法依赖）
- **优先级评分**：⭐⭐⭐⭐⭐（Agent 系统生产化必须项，解耦 Skill 与 Agent 部署）
- **评估依据**：当前 Skill 以 Markdown 文件形式存在，Agent 无法直接调用；本 Skill 是从「教学演示」到「生产可用」的关键工程基础设施
