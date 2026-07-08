---
name: scheme-b-user-feedback-evolution
description: Skill 用户反馈驱动进化方案文档。基于 pi-autoresearch 量化迭代架构，通过 GA4 用户行为信号（阅读时长/代码复制率/跳出率）+ CI 管道 + MAD 置信度检验，实现 Skill 被用户使用时自动触发质量改进循环。触发场景：新 Skill 上线后的持续优化、用户反馈驱动的迭代、高跳出率 Skill 的自动诊断修复。
---

# 方案 B：用户反馈驱动进化管道

> **定位**：外部反馈驱动进化——用户行为暴露质量问题，CI 管道自动触发改进  
> **来源**：迁移自 [davebcn87/pi-autoresearch](https://github.com/davebcn87/pi-autoresearch) + GA4 用户行为信号  
> **核心机制**：量化指标（METRIC）+ MAD 置信度过滤 + git 原子提交/回滚

---

## 一、核心理念

```
用户访问 Skill 页面
    ↓
GA4 收集行为信号（停留/复制/评分/跳出）
    ↓
每日聚合 → 低分 Skill 加入改进队列
    ↓
自动触发 .auto/measure.sh 计算质量指标
    ↓
LLM 改进 → 重测 → keep/discard（MAD 置信度）
    ↓
.auto/skill_logs/{skill_id}.jsonl（永久记录）
    ↓
飞书推送日报
```

**与方案 A 的本质区别**：
- 方案 A：不管用户行为，从内部发现问题
- 方案 B：由用户行为数据定向触发，改的是**用户觉得差的**那些维度

---

## 二、用户行为信号设计

### GA4 自定义事件（已在 Skill 页面埋点）

| 事件名 | 触发条件 | 质量意义 |
|--------|---------|---------|
| `skill_view` | 页面加载 | 基础流量 |
| `skill_code_copy` | 点击复制代码按钮 | 代码实用性 ↑ |
| `skill_time_deep` | 停留 > 120s | 内容深度 ↑ |
| `skill_time_shallow` | 停留 < 15s 即跳出 | 质量问题 ↓ |
| `skill_link_click` | 点击[[关联Skill]] | 图谱价值 ↑ |

### 质量分计算（用户信号版）

```python
def compute_user_score(skill_id: str, window_days: int = 14) -> float:
    """
    基于 GA4 14天滚动窗口计算用户感知质量分
    分数范围 0-10
    """
    ga4_data = fetch_ga4_events(skill_id, window_days)
    
    views = ga4_data.get("skill_view", 1)
    copies = ga4_data.get("skill_code_copy", 0)
    deep_reads = ga4_data.get("skill_time_deep", 0)
    bounces = ga4_data.get("skill_time_shallow", 0)
    link_clicks = ga4_data.get("skill_link_click", 0)
    
    # 复制率：代码质量代理指标
    copy_rate = copies / views
    # 深读率：内容深度代理指标  
    deep_rate = deep_reads / views
    # 跳出率：整体质量反指标
    bounce_rate = bounces / views
    # 关联点击率：图谱价值代理指标
    link_rate = link_clicks / views
    
    score = (
        copy_rate * 3.0      # 代码被复制 = 最强正信号（权重最大）
        + deep_rate * 2.5    # 深度阅读 = 内容有价值
        + link_rate * 1.5    # 关联跳转 = 图谱有效
        - bounce_rate * 2.0  # 快速跳出 = 质量问题（负向）
        + 3.0                # 基础分（确保新 Skill 也有分数）
    )
    
    return min(max(round(score, 2), 0), 10)
```

---

## 三、量化度量脚本

```bash
#!/bin/bash
# .auto/measure_skill_quality.sh <skill_file> [skill_id]
set -euo pipefail

SKILL="$1"
SKILL_ID="${2:-$(basename "$SKILL" .md)}"

# ── 1. 结构维度（纯文本分析）──────────────────────────────

MODULES=$(python3 -c "
import re; c=open('$SKILL',encoding='utf-8').read()
print(sum(1 for h in ['## ①','## ②','## ③','## ④','## ⑤'] if h in c))
")

HAS_PAPER=$(python3 -c "
import re; c=open('$SKILL',encoding='utf-8').read()
print(1 if re.search(r'arXiv|NeurIPS|ICML|KDD|ICLR', c) else 0)
")

GRAPH_LINKS=$(python3 -c "
import re; c=open('$SKILL',encoding='utf-8').read()
print(len(re.findall(r'\[\[Skill-[^\]]+\]\]', c)))
")

# ── 2. 代码可执行性 ──────────────────────────────────────

TMPCODE=$(mktemp /tmp/skill_XXXXXX.py)
CODE_SCORE=0
python3 -c "
import re
c=open('$SKILL',encoding='utf-8').read()
blocks=re.findall(r'\x60\x60\x60python\n(.*?)\x60\x60\x60', c, re.DOTALL)
if blocks:
    open('$TMPCODE', 'w', encoding='utf-8').write(blocks[0])
" 2>/dev/null

if [ -s "$TMPCODE" ]; then
    if python3 -m py_compile "$TMPCODE" 2>/dev/null; then
        # 语法OK，尝试运行
        if timeout 30s python3 "$TMPCODE" 2>/dev/null | grep -q '\[✓\]'; then
            CODE_SCORE=10
        elif timeout 30s python3 "$TMPCODE" 2>/dev/null; then
            CODE_SCORE=6
        else
            CODE_SCORE=2  # 语法OK但运行失败
        fi
    else
        CODE_SCORE=1  # 语法错误
    fi
fi
rm -f "$TMPCODE"

# ── 3. LLM 内容质量（可选，需要 API key）────────────────

LLM_SCORE=5.0
if [ -n "${DEEPSEEK_API_KEY:-}" ]; then
    LLM_SCORE=$(python3 -c "
import requests, json, os
content = open('$SKILL', encoding='utf-8').read()[:3000]
try:
    resp = requests.post('https://api.deepseek.com/chat/completions',
        headers={'Authorization': 'Bearer ' + os.environ['DEEPSEEK_API_KEY']},
        json={'model': 'deepseek-chat', 'messages': [{
            'role': 'user',
            'content': '''评估这个Skill的质量（只看内容质量，不看格式）：
- 代码是否实现了真实算法（0-3分）
- 业务场景是否具体（有品类/数字）（0-3分）
- 算法原理是否有数学直觉（0-2分）
- 是否有跨学科视角（0-2分）
只输出总分（0-10的小数），不要解释：
---
''' + content}],
        'max_tokens': 10, 'temperature': 0.1},
        timeout=20)
    result = resp.json()['choices'][0]['message']['content'].strip()
    score = float(''.join(c for c in result if c.isdigit() or c == '.'))
    print(min(max(score, 0), 10))
except:
    print(5.0)
" 2>/dev/null || echo "5.0")
fi

# ── 4. 综合评分输出 ──────────────────────────────────────

TOTAL=$(python3 -c "
m = min(int('$MODULES'), 5) * 0.4   # 最多2.0
p = int('$HAS_PAPER') * 1.5
g = min(int('$GRAPH_LINKS'), 5) * 0.3  # 最多1.5
c = float('$CODE_SCORE') * 0.4     # 代码满分4.0
l = float('$LLM_SCORE')            # LLM满分10.0，按比例
total = m + p + g + c + l
print(round(min(total, 10), 2))
")

# 输出 METRIC（pi-autoresearch 格式）
echo "METRIC skill_score=$TOTAL"
echo "METRIC modules=$MODULES"
echo "METRIC has_paper=$HAS_PAPER"
echo "METRIC graph_links=$GRAPH_LINKS"
echo "METRIC code_score=$CODE_SCORE"
echo "METRIC llm_score=$LLM_SCORE"
```

---

## 四、自动改进管道（Python）

```python
# .auto/auto_improve_pipeline.py
"""
pi-autoresearch style Skill 改进管道
每次改进 = 一次 git commit 或完整回滚
使用 MAD 置信度过滤噪声改进
"""
import json, subprocess, statistics, math
from pathlib import Path
from datetime import datetime
from typing import Optional

LOG_DIR = Path(".auto/skill_logs")
TARGET_SCORE = 8.0
MAX_ROUNDS = 3


def run_measurement(skill_path: str) -> dict:
    """运行质量度量脚本，解析 METRIC 输出"""
    result = subprocess.run(
        [".auto/measure_skill_quality.sh", skill_path],
        capture_output=True, text=True, timeout=60
    )
    metrics = {}
    for line in result.stdout.strip().split('\n'):
        if line.startswith('METRIC '):
            parts = line[7:].split('=', 1)
            if len(parts) == 2:
                try:
                    metrics[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return metrics


def compute_mad_confidence(history: list[float], new_value: float, baseline: float) -> Optional[float]:
    """
    MAD 置信度（来自 pi-autoresearch）
    置信度 = |改进量| / MAD
    ≥2.0 = 绿（真实改进）
    1.0-2.0 = 黄（边缘）
    <1.0 = 红（噪声内）
    """
    if len(history) < 3:
        return None
    
    median = statistics.median(history)
    deviations = [abs(v - median) for v in history]
    mad = statistics.median(deviations)
    
    if mad == 0:
        return None
    
    delta = abs(new_value - baseline)
    return delta / mad


def auto_improve_skill(skill_path: str, target_score: float = TARGET_SCORE) -> dict:
    """
    对单个 Skill 运行 pi-autoresearch 风格的改进循环
    """
    skill_id = Path(skill_path).stem
    log_file = LOG_DIR / f"{skill_id}.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 测量基线
    baseline_metrics = run_measurement(skill_path)
    baseline_score = baseline_metrics.get("skill_score", 0)
    
    print(f"📊 {skill_id}: 基线={baseline_score}/10")
    
    if baseline_score >= target_score:
        print(f"   ✅ 已达到目标，跳过")
        return {"skill_id": skill_id, "status": "skip", "score": baseline_score}
    
    history = [baseline_score]
    current_best = baseline_score
    results = []
    
    for round_n in range(MAX_ROUNDS):
        # 调用 LLM 改进（针对最弱指标）
        weakest = _find_weakest_metric(baseline_metrics)
        improved_content = _llm_improve(skill_path, weakest, round_n)
        
        # 写入改进版本
        original = Path(skill_path).read_text(encoding='utf-8')
        Path(skill_path).write_text(improved_content, encoding='utf-8')
        
        # 测量新版本
        new_metrics = run_measurement(skill_path)
        new_score = new_metrics.get("skill_score", 0)
        history.append(new_score)
        
        # MAD 置信度检验
        confidence = compute_mad_confidence(history, new_score, baseline_score)
        
        # keep/discard 决策
        if new_score > current_best and (confidence is None or confidence >= 1.0):
            current_best = new_score
            status = "keep"
            subprocess.run(["git", "add", skill_path], capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", 
                 f"evolve({skill_id}): round {round_n+1}, score {baseline_score:.1f}→{new_score:.1f}"],
                capture_output=True
            )
        else:
            # 回滚到最佳版本
            Path(skill_path).write_text(original, encoding='utf-8')
            status = "discard"
        
        log_entry = {
            "round": round_n, "status": status,
            "score": new_score, "best": current_best,
            "confidence": confidence, "weakest": weakest,
            "timestamp": datetime.now().isoformat()
        }
        results.append(log_entry)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        
        conf_icon = "🟢" if (confidence or 0) >= 2 else ("🟡" if (confidence or 0) >= 1 else "🔴")
        print(f"   Round {round_n+1}: {status} {new_score:.1f} {conf_icon}conf={confidence:.1f if confidence else 'N/A'}")
        
        if current_best >= target_score:
            break
    
    return {
        "skill_id": skill_id,
        "initial_score": baseline_score,
        "final_score": current_best,
        "delta": current_best - baseline_score,
        "rounds": results
    }


def _find_weakest_metric(metrics: dict) -> str:
    """找最弱维度"""
    priority = {
        "has_paper": (1 - metrics.get("has_paper", 0)) * 10,    # 无论文=最高优先
        "code_score": (10 - metrics.get("code_score", 0)) / 10 * 8,  # 代码分低
        "llm_score": (10 - metrics.get("llm_score", 5)) / 10 * 7,   # LLM分低
    }
    return max(priority, key=lambda k: priority[k])


def _llm_improve(skill_path: str, weakest: str, round_n: int) -> str:
    """调用 DeepSeek V4 Pro 改进 Skill"""
    import requests, os
    
    content = Path(skill_path).read_text(encoding='utf-8', errors='replace')
    
    improve_instructions = {
        "has_paper": "在 ## ① 和 frontmatter 中明确加入论文引用（arXiv ID 或顶会论文名）",
        "code_score": "修复代码使其可运行，确保末尾有 print('[✓] 测试通过')",
        "llm_score": "增加代码算法深度、丰富业务场景数字、补充三轨验证"
    }
    
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return content
    
    try:
        resp = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "deepseek-chat",
                "messages": [{
                    "role": "user",
                    "content": f"改进这个 Skill 卡片（第{round_n+1}轮），重点改进：{improve_instructions[weakest]}\n\n只输出改进后的完整 markdown，不要解释：\n\n{content}"
                }],
                "max_tokens": 4000,
                "temperature": 0.3
            },
            timeout=60
        )
        return resp.json()["choices"][0]["message"]["content"]
    except Exception:
        return content
```

---

## 五、与 GA4 的集成（触发条件）

```python
# .auto/ga4_trigger.py
"""每日 cron 触发：从 GA4 找低分 Skill，加入改进队列"""

def daily_trigger():
    """每天 02:00 运行，找过去 7 天用户体验差的 Skill"""
    low_score_skills = fetch_low_engagement_skills(
        min_views=10,        # 至少有10次访问才有统计意义
        max_copy_rate=0.05,  # 代码复制率 < 5%（用户不觉得代码有用）
        min_bounce_rate=0.7, # 跳出率 > 70%（用户不愿意深读）
        limit=20             # 每天最多处理 20 个
    )
    
    for skill_id in low_score_skills:
        skill_path = find_skill_path(skill_id)
        auto_improve_skill(str(skill_path), target_score=8.0)
    
    # 飞书推送日报
    notify_feishu_daily_report(low_score_skills)
```

---

## 六、飞书日报格式

每日改进完成后推送到飞书：

```
📊 Skill 进化日报 — 2026-07-04

今日改进 8 个 Skill
✅ 成功提升: 6 个（平均 +2.3分）
↩️ 回滚: 2 个（噪声内，无统计显著性）

Top 改进:
• Skill-Market-Size-Estimation: 6.2 → 8.5 (+2.3) 🟢
• Skill-LTV-Prediction-ZILN: 5.8 → 7.9 (+2.1) 🟢
• Skill-Demand-Forecasting: 7.1 → 8.0 (+0.9) 🟡

低质量队列（明日处理）:
• Skill-ABC（跳出率 82%，复制率 2%）
• Skill-XYZ（停留时间均值 11s）

查看全部日志: /skills/agent-report.html
```
