# Skill Card: Click Fraud Detection（广告刷量检测）

> **领域**: 19-风控反欺诈 | **类型**: 综合萃取

---

## ① 算法原理

检测广告点击中的无效流量（IVT）——Bot 点击、竞品恶意点击、重复点击。用时间序列异常 + 行为模式识别。

**检测信号**：
- 点击间隔 < 1 秒（人类不可能）
- 同一 IP/DeviceID 重复点击 > 10 次/天
- 点击后 0 秒跳出（bounce rate 100%）
- CTR 异常飙升（CTR 突然 3σ 偏离均值）

**集成检测**：加权投票——任一信号触发概率 > 0.7 → 标记为疑似 IVT。

---

## ② 母婴出海应用案例

FB 吸奶器广告突然 CTR 从 2.1% 飙升到 8.5%，但转化率从 3% 降到 0.1%。检测到 85% 点击来自 3 个 IP 段且 99% bounce。标记为 IVT 攻击→向 FB 申请退款 $2,400。后续部署实时 IVT 过滤，月减少无效花费 $500-1,000。

年化：**6-15 万元**。

---

## ③ 代码模板

```python
import numpy as np

def detect_click_fraud(clicks_per_hour, ctr_history, bounce_rates, ip_freq):
    """多信号集成 IVT 检测"""
    signals = []
    # 信号1: CTR spike
    z_ctr = (clicks_per_hour['ctr'] - np.mean(ctr_history)) / max(np.std(ctr_history), 0.001)
    signals.append(min(abs(z_ctr)/4, 1.0))
    # 信号2: bounce rate
    signals.append(bounce_rates.get('avg', 0))
    # 信号3: IP concentration
    signals.append(min(ip_freq.get('top3_ratio', 0), 1.0))
    
    ivt_risk = np.mean(signals)
    return {'ivt_risk': ivt_risk, 'is_ivt': ivt_risk > 0.7, 'signals': signals}

# test
r = detect_click_fraud(
    {'ctr': 0.085}, [0.02]*30, {'avg': 0.95}, {'top3_ratio': 0.85})
print(f"IVT risk: {r['ivt_risk']:.0%}, is IVT: {r['is_ivt']}")
assert r['is_ivt']
print("[✓] Click Fraud Detection 测试通过")
```

---

## ④ 技能关联

- **前置**：[[Skill-ROAS-Budget-Optimization]] | [[Skill-Time-Series-Anomaly-Detection]]
- **组合**：[[Skill-Transaction-Anomaly-Detection]]（统一风控管道）

---

## ⑤ 商业价值

- **ROI**：6-15 万元 | **难度**：⭐⭐☆☆☆ | **优先级**：⭐⭐⭐☆☆
