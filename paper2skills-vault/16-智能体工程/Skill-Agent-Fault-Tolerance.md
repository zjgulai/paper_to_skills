# Skill Card: Agent Fault Tolerance（Agent 容错回退）

> **领域**: 16-智能体工程 | **类型**: 综合萃取

---

## ① 算法原理

Agent 执行可能因 API 超时、LLM 输出格式错误、工具返回异常而失败。容错机制：(1) Retry with exponential backoff（$t_{retry} = \min(t_{base} \cdot 2^n, t_{max})$）；(2) Fallback 策略——LLM 失败切备选模型，工具失败用简化版；(3) Circuit Breaker——连续失败 N 次后停止尝试，避免雪崩。

**状态机**：`NORMAL → RETRY(3次) → FALLBACK → CIRCUIT_OPEN(冷却30s) → HALF_OPEN → NORMAL`。

---

## ② 母婴出海应用案例

WF-A 补货 Agent 调用库存 API 超时（3 次重试均失败）→ Circuit Breaker 打开→降级为"基于最新已知库存+需求预测"的保守建议 → 30 秒后半开测试→API 恢复→恢复正常。避免因 API 抖动导致补货决策完全中断。

---

## ③ 代码模板

```python
import time, random

class CircuitBreaker:
    def __init__(self, failure_threshold=3, cooldown_sec=30):
        self.failures = 0; self.threshold = failure_threshold
        self.cooldown = cooldown_sec; self.last_failure = 0; self.open = False
    
    def call(self, fn, *args, fallback=None, **kwargs):
        if self.open:
            if time.time() - self.last_failure > self.cooldown:
                self.open = False; self.failures = 0  # half-open
            elif fallback:
                return fallback(*args, **kwargs)
        
        for attempt in range(3):
            try:
                result = fn(*args, **kwargs)
                self.failures = 0; return result
            except Exception:
                wait = min(2**attempt, 8); time.sleep(wait * 0.01)  # scaled for test
        
        self.failures += 1
        if self.failures >= self.threshold:
            self.open = True; self.last_failure = time.time()
        return fallback(*args, **kwargs) if fallback else None

# test
cb = CircuitBreaker(failure_threshold=2, cooldown_sec=0.1)
fail_fn = lambda: (_ for _ in ()).throw(Exception("timeout"))
fallback = lambda: "cached_result"
assert cb.call(fail_fn, fallback=fallback) == "cached_result"
assert cb.open  # circuit opened after 2 failures
print("[✓] Agent Fault Tolerance 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-Agent-Safety-Guardrails]]
- **组合**：[[Skill-Cost-Aware-Agent-Scheduling]] | [[Skill-Orchestration-Trace-RL]]

---

## ⑤ 商业价值

- **ROI**：避免 Agent 中断导致的决策延迟，年化隐性 **5-15 万元**
- **难度**：⭐⭐☆☆☆ | **优先级**：⭐⭐⭐⭐☆
