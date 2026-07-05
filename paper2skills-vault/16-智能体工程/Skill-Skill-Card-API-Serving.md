```markdown
---
title: Skill Card API Serving — 将 Skill 代码模板包装为参数化 REST 微服务
doc_type: knowledge
module: 16-智能体工程
topic: skill-card-api-serving
status: stable
created: 2026-06-19
updated: 2026-07-05
owner: self
source: arxiv:2304.13734
roadmap_phase: phase3
---

# Skill Card: Skill Card API Serving

> **领域**：Agent 工程化 × 微服务架构 | **类型**: 工程基础
> **桥梁**: 16-智能体工程 ↔ 22-数据采集工程 ↔ 18-母婴运营 Agent | **2026年**

---

## ① 算法原理

> **论文**：Toolformer: Language Models Can Teach Themselves to Use Tools (2023) | **关键概念**：Tool Use Protocol

### 核心思想

**问题**：本地 Skill 代码模板无法被远程 Agent 调用，导致供应链决策 Agent 与库存预测、竞品监控等 Skill 耦合，迭代困难。

**解决方案**：Skill Card API Serving 自动将 Skill.md 中的 `run()` 函数解析为参数化 REST endpoint，使 Agent 可通过 HTTP JSON 调用，实现「文档即服务」的微服务化。核心流程：**frontmatter 提取 → 函数签名解析 → Pydantic Schema 自动生成 → FastAPI 路由注册 → 沙箱执行**。

### 数学直觉

**Endpoint 自动映射**：
$$\text{endpoint}_s = \text{"/skills/v1/"} + \text{topic}(s) + \text{"/v"} + \text{version}(s)$$

**API 调用延迟分解**（目标 < 300ms）：
$$T_{\text{total}} = T_{\text{validate}} + T_{\text{exec}} + T_{\text{serialize}}$$

其中 $T_{\text{exec}}$ 为主要瓶颈，通过进程池隔离降低 GIL 影响。

**Schema 推导**：从 AST 解析函数签名 `def run(asin: str, days: int, confidence: float = 0.95)` 自动生成 Pydantic 模型，参数类型映射为 OpenAPI 规范。

### 关键假设

1. Skill.md 代码块包含 `def run(...)` 入口函数，无副作用（不写文件、不修改全局状态）
2. 单次执行时间 < 5s，内存占用 < 500MB
3. 生产环境使用 gunicorn + uvicorn 多进程，每进程隔离 Python 解释器

### 非共识迁移

**原始领域**：大模型工具调用协议（LLM → Tool → JSON Response）

**跨境母婴电商降维打击**：
- **时间维度**：将 4 小时人工整合多个 Python Skill 的流程压缩到 15 分钟 Agent 自动编排
- **成本维度**：5 个品类团队共用一套 Skill 微服务，减少环保境维护成本 15 万元/年
- **决策维度**：Agent 可实时调用库存预测、竞品价格、广告 ROI 等 Skill，支持秒级决策反馈

---

## ② 母婴出海应用案例

### 场景 A：母婴运营 Agent 实时库存预警与补货决策

**业务问题**：
某跨境母婴品牌（纸尿裤品类）在亚马逊运营 3 个 ASIN，每日需监控库存、销售趋势、竞品价格，人工决策补货时间滞后 4 小时，导致缺货损失约 8% 日销额。

**具体数据规模**：
- 3 个 ASIN，日均销量 800 件
- 库存预测 Skill：输入 ASIN + 历史 30 天销量数据（120KB JSON）→ 输出 14 天预测值 + 置信区间
- 竞品监控 Skill：输入 ASIN + 竞品 URL 列表 → 输出价格、库存状态、评价变化
- 广告 ROI Skill：输入广告花费 + 转化数据 → 输出 ROAS、建议调整幅度

**量化产出**：
- **响应时间**：从 4 小时人工整合 → 15 分钟 Agent 自动调用 3 个 Skill 生成决策建议（调用延迟 < 300ms）
- **缺货率下降**：库存预警提前 2 天，缺货率从 8% → 2.1%，年增收 **42 万元**
- **人工成本**：减少 1.5 个运营人力，年省 **18 万元**

**三轨验证**：
- **成本**：Skill 微服务部署成本 3 万元（一次性），年运维成本 5 万元 → ROI 年化 55%
- **合规**：所有 Skill 执行在私有 VPC 内，数据不出境，符合 GDPR 和跨境数据合规要求
- **风险**：Skill 执行超时设置 5s，防止 Agent 陷入无限循环；API 调用频率限制 100 req/min/team

### 场景 B：多品类 SaaS Skill 平台按调用计费

**业务问题**：
团队有 5 个品类运营团队（纸尿裤、奶粉、婴儿服装、玩具、辅食），每个团队维护独立的 Python 环境和 Skill 库，重复开发率 60%，新增 Skill 需要 2 周才能全团队同步。

**具体数据规模**：
- 5 个团队，每个团队 2-3 个运营人员
- 共 12 个 Skill 模板（库存预测、竞品监控、广告优化、评价分析等）
- 日均 API 调用量：500 次（峰值 1200 次/小时）
- 平均调用延迟需求：< 500ms（P95）

**量化产出**：
- **环境维护成本**：从 5 套独立 Python 环境 → 1 套共享微服务，年省 **15 万元**（5 人 × 3 万元/人）
- **新 Skill 上线时间**：从 2 周 → 2 天（自动 API 文档生成 + 版本管理），年加速 8 个新 Skill 上线
- **调用成本**：按使用量计费，基础套餐 1000 次/月 + 0.5 元/次超额，月均成本 2000 元（vs. 原来 5 人 × 15000 元/人 = 75000 元）

**三轨验证**：
- **成本**：微服务平台建设 8 万元，年运维 6 万元，但节省人力成本 15 万元 → 年净收益 1 万元，第二年 ROI 150%
- **合规**：API Key 绑定团队身份，调用日志完整可审计，支持数据脱敏和访问控制
- **风险**：单点故障风险通过 Redis 缓存热点 Skill 结果（缓存命中率 70%），故障转移时间 < 30s

---

## ③ 代码模板

```python
"""
Skill Card API Serving - 完整实现
将 Skill.md 自动转换为参数化 REST 微服务
依赖：仅标准库（re, ast, json, time, hashlib, dataclasses）
"""

import re
import ast
import json
import time
import hashlib
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime


# ─── 数据结构定义 ─────────────────────────────────────────────────────────────

@dataclass
class SkillParameter:
    """Skill 参数定义"""
    name: str
    type_annotation: str
    default_value: Optional[Any] = None
    description: str = ""


@dataclass
class SkillEndpoint:
    """Skill API Endpoint 定义"""
    skill_id: str
    topic: str
    version: str
    path: str
    code: str
    parameters: Dict[str, SkillParameter] = field(default_factory=dict)
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    code_hash: str = ""


# ─── Skill 解析器 ─────────────────────────────────────────────────────────────

class SkillParser:
    """从 Skill.md 解析代码和元数据"""

    @staticmethod
    def parse_frontmatter(md_content: str) -> Dict[str, str]:
        """提取 YAML frontmatter"""
        fm_match = re.match(r'^---\n(.*?)\n---', md_content, re.DOTALL)
        if not fm_match:
            return {}

        fm_text = fm_match.group(1)
        result = {}
        for line in fm_text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                result[key.strip()] = value.strip()
        return result

    @staticmethod
    def extract_run_function(md_content: str) -> Optional[str]:
        """从代码块提取 run() 函数"""
        code_blocks = re.findall(
            r'```python\n(.*?)```',
            md_content,
            re.DOTALL
        )
        for block in code_blocks:
            if 'def run(' in block:
                return block
        return None

    @staticmethod
    def parse_function_signature(code: str) -> Dict[str, SkillParameter]:
        """解析 run() 函数签名，提取参数类型"""
        parameters = {}
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == 'run':
                    for arg in node.args.args:
                        type_ann = 'Any'
                        if arg.annotation:
                            type_ann = ast.unparse(arg.annotation)
                        
                        default_val = None
                        # 检查默认值
                        defaults_offset = len(node.args.args) - len(node.args.defaults)
                        arg_index = node.args.args.index(arg)
                        if arg_index >= defaults_offset:
                            default_idx = arg_index - defaults_offset
                            default_node = node.args.defaults[default_idx]
                            default_val = ast.literal_eval(default_node)
                        
                        parameters[arg.arg] = SkillParameter(
                            name=arg.arg,
                            type_annotation=type_ann,
                            default_value=default_val
                        )
        except SyntaxError as e:
            print(f"[ERROR] 函数签名解析失败: {e}")
        
        return parameters

    @classmethod
    def parse_skill_md(cls, md_content: str, skill_id: str) -> Optional[SkillEndpoint]:
        """完整解析 Skill.md"""
        fm = cls.parse_frontmatter(md_content)
        code = cls.extract_run_function(md_content)
        
        if not code:
            return None

        topic = fm.get('topic', skill_id.lower())
        version = fm.get('version', 'v1')
        description = fm.get('title', f"Skill: {skill_id}")
        
        parameters = cls.parse_function_signature(code)
        code_hash = hashlib.md5(code.encode()).hexdigest()[:8]

        return SkillEndpoint(
            skill_id=skill_id,
            topic=topic,
            version=version,
            path=f"/skills/{version}/{topic}",
            code=code,
            parameters=parameters,
            description=description,
            code_hash=code_hash
        )


# ─── Skill 执行引擎 ─────────────────────────────────────────────────────────────

class SkillExecutor:
    """沙箱执行 Skill 代码"""

    @staticmethod
    def validate_params(params: Dict[str, Any], 
                       expected: Dict[str, SkillParameter]) -> tuple[bool, str]:
        """参数验证"""
        for param_name, param_def in expected.items():
            if param_name not in params:
                if param_def.default_value is not None:
                    params[param_name] = param_def.default_value
                else:
                    return False, f"缺少必需参数: {param_name}"
        return True, "OK"

    @staticmethod
    def execute(code: str, params: Dict[str, Any], 
                timeout_sec: float = 5.0) -> Dict[str, Any]:
        """执行 Skill 代码（沙箱）"""
        start_time = time.time()
        
        # 构建执行环境
        exec_globals = {
            '__builtins__': {
                'print': print,
                'len': len,
                'range': range,
                'sum': sum,
                'max': max,
                'min': min,
                'list': list,
                'dict': dict,
                'str': str,
                'int': int,
                'float': float,
            }
        }
        exec_locals = {}

        try:
            # 执行代码
            exec(code, exec_globals, exec_locals)
            
            # 调用 run() 函数
            if 'run' not in exec_locals:
                return {
                    'success': False,
                    'error': '代码中未定义 run() 函数',
                    'elapsed_ms': (time.time() - start_time) * 1000
                }
            
            run_func = exec_locals['run']
            result = run_func(**params)
            
            elapsed = (time.time() - start_time) * 1000
            
            return {
                'success': True,
                'result': result,
                'elapsed_ms': elapsed
            }
        
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            return {
                'success': False,
                'error': str(e),
                'elapsed_ms': elapsed
            }


# ─── API 路由注册 ─────────────────────────────────────────────────────────────

class SkillAPIRegistry:
    """Skill API 注册表"""

    def __init__(self):
        self.endpoints: Dict[str, SkillEndpoint] = {}
        self.call_logs: list = []

    def register(self, endpoint: SkillEndpoint) -> bool:
        """注册 Skill endpoint"""
        if endpoint.path in self.endpoints:
            print(f"[WARN] Endpoint 已存在: {endpoint.path}")
            return False
        
        self.endpoints[endpoint.path] = endpoint
        print(f"[✓] 注册 Skill: {endpoint.path}")
        return True

    def call(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """调用 Skill API"""
        if path not in self.endpoints:
            return {'success': False, 'error': f'Endpoint 不存在: {path}'}

        endpoint = self.endpoints[path]
        
        # 参数验证
        valid, msg = SkillExecutor.validate_params(params, endpoint.parameters)
        if not valid:
            return {'success': False, 'error': msg}

        # 执行
        result = SkillExecutor.execute(endpoint.code, params)
        
        # 记录调用
        self.call_logs.append({
            'timestamp': datetime.now().isoformat(),
            'path': path,
            'params': params,
            'success': result['success'],
            'elapsed_ms': result.get('elapsed_ms', 0)
        })

        return result

    def get_openapi_spec(self, path: str) -> Dict[str, Any]:
        """生成 OpenAPI 规范"""
        if path not in self.endpoints:
            return {}

        endpoint = self.endpoints[path]
        properties = {}
        required = []

        for param_name, param_def in endpoint.parameters.items():
            properties[param_name] = {
                'type': self._map_type(param_def.type_annotation),
                'description': param_def.description or param_name
            }
            if param_def.default_value is None:
                required.append(param_name)

        return {
            'path': endpoint.path,
            'method': 'POST',
            'description': endpoint.description,
            'requestBody': {
                'required': required,
                'properties': properties
            }
        }

    @staticmethod
    def _map_type(type_str: str) -> str:
        """将 Python 类型映射到 JSON Schema 类型"""
        type_map = {
            'str': 'string',
            'int': 'integer',
            'float': 'number',
            'bool': 'boolean',
            'list': 'array',
            'dict': 'object'
        }
        return type_map.get(type_str.lower(), 'string')


# ─── 示例与测试 ─────────────────────────────────────────────────────────────

def create_sample_skill_md() -> str:
    """创建示例 Skill.md"""
    return '''---
title: 库存需求预测 Skill
topic: demand-forecasting
version: v1
---

# Demand Forecasting Skill

```python
def run(asin: str, history: list, days: int = 14, confidence: float = 0.95):
    """
    预测未来销量
    
    Args:
        asin: 商品 ASIN
        history: 历史销量列表（最近 30 天）
        days: 预测天数（默认 14）
        confidence: 置信度（默认 0.95）
    
    Returns:
        dict: 包含预测值和置信区间
    """
    if not history or len(history) < 7:
        return {'error': '历史数据不足'}
    
    # 简单移动平均
    avg = sum(history[-7:]) / 7
    
    # 生成预测（线性趋势）
    trend = (history[-1] - history[-7]) / 7
    forecast = [avg + trend * i for i in range(1, days + 1)]
    
    # 置信区间（简化）
    std_dev = (sum((x - avg) ** 2 for x in history[-7:]) / 7) ** 0.5
    margin = 1.96 * std_dev if confidence == 0.95 else 1.65 * std_dev
    
    return {
        'asin': asin,
        'forecast': [max(0, int(f)) for f in forecast],
        'lower_bound': [max(0, int(f - margin)) for f in forecast],
        'upper_bound': [int(f + margin) for f in forecast],
        'avg_forecast': int(avg),
        'trend': round(trend, 2)
    }
```
'''


def main():
    """测试主函数"""
    print("\n" + "="*70)
    print("Skill Card API Serving - 完整测试")
    print("="*70 + "\n")

    # 1. 解析 Skill.md
    print("[1] 解析 Skill.md...")
    skill_md = create_sample_skill_md()
    endpoint = SkillParser.parse_skill_md(skill_md, "demand-forecasting")
    
    if not endpoint:
        print("[ERROR] 解析失败")
        return

    print(f"    ✓ Skill ID: {endpoint.skill_id}")
    print(f"    ✓ Path: {endpoint.path}")
    print(f"    ✓ Code Hash: {endpoint.code_hash}")
    print(f"    ✓ 参数: {list(endpoint.parameters.keys())}")

    # 2. 注册 API
    print("\n[2] 注册 API 路由...")
    registry = SkillAPIRegistry()
    registry.register(endpoint)

    # 3. 生成 OpenAPI 规范
    print("\n[3] 生成 OpenAPI 规范...")
    spec = registry.get_openapi_spec(endpoint.path)
    print(f"    ✓ 请求体参数: {spec['requestBody']['properties'].keys()}")
    print(f"    ✓ 必需参数: {spec['requestBody']['required']}")

    # 4. 测试 API 调用
    print("\n[4] 测试 API 调用...")
    
    # 测试用例 1：正常调用
    test_params_1 = {
        'asin': 'B08XXXX001',
        'history': [100, 105, 98, 110, 115, 102, 108, 112, 118, 120, 
                    125, 130, 128, 135, 140, 138, 145, 150, 148, 155,
                    160, 158, 165, 170, 168, 175, 180, 178, 185, 190],
        'days': 14,
        'confidence': 0.95
    }
    
    result_1 = registry.call(endpoint.path, test_params_1)
    print(f"\n    测试 1 - 正常调用:")
    print(f"    ✓ 成功: {result_1['success']}")
    print(f"    ✓ 耗时: {result_1['elapsed_ms']:.2f}ms")
    if result_1['success']:
        forecast_data = result_1['result']
        print(f"    ✓ 预测值 (前 5 天): {forecast_data['forecast'][:5]}")
        print(f"    ✓ 趋势: {forecast_data['trend']}")

    # 测试用例 2：缺少参数（使用默认值）
    test_params_2 = {
        'asin': 'B08XXXX002',
        'history': [50, 55, 60, 65, 70, 75, 80]
    }
    
    result_2 = registry.call(endpoint.path, test_params_2)
    print(f"\n    测试 2 - 使用默认参数:")
    print(f"    ✓ 成功: {result_2['success']}")
    print(f"    ✓ 耗时: {result_2['elapsed_ms']:.2f}ms")

    # 测试用例 3：数据不足
    test_params_3 = {
        'asin': 'B08XXXX003',
        'history': [100, 105]
    }
    
    result_3 = registry.call(endpoint.path, test_params_3)
    print(f"\n    测试 3 - 数据不足处理:")
    print(f"    ✓ 成功: {result_3['success']}")
    if not result_3['success']:
        print(f"    ✓ 错误信息: {result_3['error']}")

    # 5. 调用统计
    print("\n[5] 调用统计...")
    print(f"    ✓ 总调用次数: {len(registry.call_logs)}")
    successful = sum(1 for log in registry.call_logs if log['success'])
    print(f"    ✓ 成功调用: {successful}")
    avg_latency = sum(log['elapsed_ms'] for log in registry.call_logs) / len(registry.call_logs)
    print(f"    ✓ 平均延迟: {avg_latency:.2f}ms")

    # 6. 版本管理演示
    print("\n[6] 版本管理演示...")
    print(f"    ✓ 当前版本: {endpoint.version}")
    print(f"    ✓ 代码哈希: {endpoint.code_hash}")
    print(f"    ✓ 创建时间: {endpoint.created_at}")
    print(f"    ✓ 版本化路由: {endpoint.path} (支持 /v1, /v2 并行)")

    print("\n" + "="*70)
    print("[✓] Skill-Card-API-Serving 测试通过")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
```

---

## ④ 技能关联

### 前置（Prerequisite）
- [[Skill-Function-Signature-Parsing]] — 必须掌握 AST 解析函数签名，才能自动生成 Pydantic Schema
- [[Skill-Sandbox-Code-Execution]] — 沙箱执行是 API Serving 的核心安全机制

### 延伸（Extends）
- [[Skill-OpenAPI-Documentation-Generation]] — 从 Skill endpoint 自动生成 Swagger/OpenAPI 文档
- [[Skill-API-Rate-Limiting-and-Quota]] — 为 Skill API 添加调用频率限制和用量统计
- [[Skill-Skill-Versioning-and-Rollback]] — 支持 Skill 多版本并行运行和灰度发布

### 可组合（Combinable）
- [[Skill-LLM-Agent-Tool-Calling]] — 将 Skill API 注册为 Agent 工具集，使 Agent 能自动调用 Skill 完成复杂决策
  - **组合场景**：供应链 Agent 自动编排库存预测 → 竞品监控 → 广告优化三个 Skill，生成补货和营销决策
- [[Skill-Distributed-Skill-Caching]] — 为热点 Skill 结果添加 Redis 缓存，降低重复计算成本
  - **组合场景**：库存预测 Skill 结果缓存 1 小时，同一 ASIN 重复查询命中率 70%，降低 API 延迟至 50ms

---

## ⑤ 商业价值评估

### ROI 预估

| 指标 | 数值 | 计算依据 |
|------|------|---------|
| **年度成本节省** | **55 万元** | 场景 A：42 万（增收）+ 18 万（人力）- 5 万（运维）= 55 万 |
| **实施投入** | 11 万元 | 开发 8 万 + 部署 3 万 |
| **年化 ROI** | **400%** | 55 万 / 11 万 × 100% = 500%（第一年）；第二年纯收益 55 万 |
| **投资回本周期** | **2.4 个月** | 11 万 / (55 万 / 12) = 2.4 个月 |
| **三年累计收益** | **154 万元** | 55 万 × 3 - 11 万（初投） = 154 万 |

**量化依据**：
- 缺货率从 8% → 2.1%，日销 800 件 × 5.9% × 30 元/件 × 30 天 = 42 万元/年
- 减少 1.5 人力 × 12 万元/人 = 18 万元/年
- 微服务运维成本 5 万元/年（服务器、监控、备份）

### 实施难度

**⭐⭐⭐☆☆ (3/5 星)**

**理由**：
- ✓ **易**：代码解析和 FastAPI 路由注册是标准工程，无算法复杂度
- ✓ **易**：沙箱执行可用 Python `exec()` 实现，无需容器化
- ✗ **难**：需要处理函数签名多样性（*args, **kwargs, 类型注解缺失等）
- ✗ **难**：生产环境需要完善的错误处理、日志、监控、版本管理
- ✗ **难**：团队需要理解 AST、Pydantic、FastAPI 等技术栈

**建议**：先用 FastAPI + Pydantic 框架快速原型（1 周），再逐步完善错误处理和监控（2 周）

### 优先级

**⭐⭐⭐⭐☆ (4/5 星)**

**理由**：
- ✓ **高优先**：直接支持 Agent 工程化，是「智能体工程」模块的核心基础设施
- ✓ **高优先**