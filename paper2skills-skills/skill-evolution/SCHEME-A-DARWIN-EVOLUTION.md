---
name: scheme-a-darwin-evolution
description: Skill 达尔文自进化引擎方案文档。基于 Darwin Skill 2.0 架构，通过 9 维度独立评委评分、变异算子、棘轮选择机制，对每个 Skill 进行批量内部自进化迭代，平均质量提升 +11.9 分。触发场景：批量存量 Skill 质量提升、季度大版本迭代、领域专项精炼。
---

# 方案 A：达尔文自进化引擎

> **定位**：内部自驱进化——Skill 自身不断突变、被独立评委筛选、保留更优个体  
> **来源**：迁移自 [alchaincyf/darwin-skill](https://github.com/alchaincyf/darwin-skill) + paper2skills 业务适配  
> **效果基准**：Darwin 2.0 在 5 个真实 Skill 上平均 +11.9 分（79.6 → 91.5/100）

---

## 一、核心理念

把每个 Skill 当成一个"物种"，定期经历自然选择压力：

```
当前 Skill → 独立评委评分 → 找最弱维度 → LLM 变异 → 重新评分 → 棘轮选择（只升不降）
     ↑                                                                        │
     └────────────────── 进化日志 .evo/{skill_id}.jsonl ◀────────────────────┘
```

三大创新点（来自 Darwin 2.0）：
1. **独立评委**：评分者和改写者是不同的 LLM 调用，避免自评偏差（LLM 自评准确率 46.4% → 独立评委 73.8%）
2. **棘轮机制**：分数只升不降，每轮只保留比当前最优更好的突变
3. **维度定向变异**：针对得分最低的维度精准改写，而非随机改写

---

## 二、9 维度评分矩阵（满分 100）

| ID | 维度 | 满分 | 检测方式 | 描述 |
|----|------|------|---------|------|
| D1 | 5模块完整性 | 8 | 自动 | ## ① ② ③ ④ ⑤ 全部存在 |
| D2 | Frontmatter规范 | 5 | 自动 | 含 title/doc_type/roadmap_phase |
| D3 | 论文引用 | 10 | 自动 | 明确 arXiv ID 或顶会引用 |
| D4 | ROI量化 | 8 | 自动 | ⑤ 商业价值含具体数字（万元/%） |
| D5 | 图谱关联 | 4 | 自动 | ≥2条 [[双括号]] 关联 |
| D6 | 代码可执行 | 10 | 执行 | 代码运行通过 + 末尾有 [✓] |
| D7 | 代码反映论文 | 15 | LLM | 代码实现了论文核心算法而非 mock |
| D8 | 算法深度 | 10 | LLM | 原理部分有数学直觉（公式+解释）|
| D9 | 场景具体性 | 15 | LLM | 有具体品类/数字，非泛泛而谈 |
| D10 | 三轨验证 | 10 | LLM | 成本/合规/风险三轨完整 |
| D11 | 跨学科迁移 | 5 | LLM | 非共识视角，跨领域降维打击 |

**通过线：70 分**  
**优秀线：85 分**  
**封顶目标：90 分（早停）**

---

## 三、变异算子库

| 触发维度 | 变异策略 | 具体操作 |
|---------|---------|---------|
| D3（无论文）| `fetch_paper_and_inject` | 搜索 arXiv，提取摘要，注入 ① 和 metadata |
| D6（代码失败）| `fix_code_executable` | 修复 import/语法/运行错误，确保 [✓] |
| D7（代码浅）| `rewrite_code_from_paper` | 从论文算法伪代码重写核心实现 |
| D8（原理浅）| `deepen_algorithm_section` | 补充数学公式 + 直觉解释 |
| D9（场景泛）| `enrich_business_scenario` | 换具体品类（暖奶器/婴儿车），加数字 |
| D10（缺三轨）| `add_three_track_verification` | 补充成本/合规/风险三轨对抗验证 |
| D11（无视角）| `add_cross_domain_insight` | 分析跨学科来源，写"非共识洞察" |

---

## 四、进化循环实现

```python
# paper2skills-skills/skill-evolution/darwin_evolve.py

import json, subprocess, re, ast
from pathlib import Path
from datetime import datetime
from typing import Optional

VAULT = Path("paper2skills-vault")
EVO_DIR = Path(".evo")
TARGET_SCORE = 90
MAX_ROUNDS = 5
LLM_MODEL = "deepseek-chat"  # DeepSeek V4 Pro


# ── 自动检测维度 ──────────────────────────────────────────

def score_auto(content: str) -> dict[str, float]:
    scores = {}
    
    # D1: 5模块
    modules = sum(1 for h in ["## ①","## ②","## ③","## ④","## ⑤"] if h in content)
    scores["D1"] = min(modules / 5 * 8, 8)
    
    # D2: Frontmatter
    has_fm = content.startswith("---\n")
    has_roadmap = "roadmap_phase:" in content[:500]
    scores["D2"] = 5 if (has_fm and has_roadmap) else (2 if has_fm else 0)
    
    # D3: 论文引用
    has_paper = bool(re.search(r'arXiv|arxiv|NeurIPS|ICML|KDD|ICLR|AAAI|WWW|SIGIR', content))
    scores["D3"] = 10 if has_paper else 0
    
    # D4: ROI量化
    value_section = content[content.find("## ⑤"):content.find("## ⑤")+600] if "## ⑤" in content else ""
    has_roi = bool(re.search(r'\d+\s*(?:万|%|元|亿)', value_section))
    scores["D4"] = 8 if has_roi else 0
    
    # D5: 图谱关联
    links = re.findall(r'\[\[Skill-[^\]]+\]\]', content)
    scores["D5"] = min(len(links) / 2 * 4, 4)
    
    return scores


def score_code(skill_path: Path) -> float:
    """D6: 代码执行测试"""
    content = skill_path.read_text(encoding="utf-8", errors="replace")
    blocks = re.findall(r'```python\n(.*?)```', content, re.DOTALL)
    if not blocks:
        return 0
    
    # 语法检查
    try:
        ast.parse(blocks[0])
    except SyntaxError:
        return 2  # 有代码但语法错误
    
    # 执行测试
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(blocks[0])
        tmp = f.name
    
    try:
        result = subprocess.run(
            ["python3", tmp], capture_output=True, text=True, timeout=30
        )
        has_check = "[✓]" in result.stdout or "[✔]" in result.stdout
        if result.returncode == 0 and has_check:
            return 10
        elif result.returncode == 0:
            return 7
        else:
            return 3
    except subprocess.TimeoutExpired:
        return 4
    finally:
        os.unlink(tmp)


def score_llm(content: str, paper_abstract: Optional[str] = None) -> dict[str, float]:
    """D7-D11: DeepSeek V4 Pro 独立评委评分（绝不对 Skill 本身做自评）"""
    import requests
    
    judge_prompt = f"""你是顶级 AI Skill 质量评审专家。请对以下 Skill 卡片进行客观评分。

{"论文摘要（用于对比代码实现深度）：" + paper_abstract[:500] if paper_abstract else ""}

---
{content[:4000]}
---

请严格按照以下 5 个维度评分，每个维度只输出数字：

D7（代码反映论文，满分15）：
- 15分：代码实现了论文的核心算法（有明确对应关系）
- 8分：代码有算法框架但未深入实现
- 0分：代码只是 mock 数据或随机生成

D8（算法深度，满分10）：
- 10分：① 原理部分有公式+直觉解释，跨学科来源明确
- 5分：有描述但缺少数学直觉
- 0分：只是算法名称堆砌

D9（场景具体性，满分15）：
- 15分：② 应用案例有具体品类（如暖奶器）+ 具体数字 + 三轨验证
- 8分：场景有但数字模糊
- 0分：泛泛而谈无具体场景

D10（三轨验证，满分10）：
- 10分：成本验证 + 合规验证 + 风险验证三轨完整
- 5分：有1-2轨
- 0分：无三轨验证

D11（跨学科迁移，满分5）：
- 5分：明确说明算法原始领域 + 为何能降维打击跨境电商问题
- 2分：有提及但不清晰
- 0分：无跨学科视角

请只输出 JSON，格式：
{{"D7": 数字, "D8": 数字, "D9": 数字, "D10": 数字, "D11": 数字, "weakest": "D7/D8/D9/D10/D11", "reason": "一句话说明最弱维度的具体问题"}}
"""
    
    # 实际调用 DeepSeek V4 Pro
    # 此处为框架，实际 key 从环境变量获取
    try:
        import os
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            return {"D7": 8, "D8": 5, "D9": 8, "D10": 5, "D11": 2, "weakest": "D10", "reason": "API key 未配置，使用默认分"}
        
        resp = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": judge_prompt}],
                "max_tokens": 200,
                "temperature": 0.1,
                "response_format": {"type": "json_object"}
            },
            timeout=30
        )
        result = json.loads(resp.json()["choices"][0]["message"]["content"])
        return result
    except Exception as e:
        return {"D7": 8, "D8": 5, "D9": 8, "D10": 5, "D11": 2, "weakest": "D10", "reason": f"评分失败: {e}"}


def evaluate_skill(skill_path: Path) -> dict:
    """完整评分（5个自动 + 1个执行 + 5个LLM）"""
    content = skill_path.read_text(encoding="utf-8", errors="replace")
    
    auto = score_auto(content)
    code = score_code(skill_path)
    llm = score_llm(content)
    
    breakdown = {**auto, "D6": code}
    for k in ["D7", "D8", "D9", "D10", "D11"]:
        breakdown[k] = float(llm.get(k, 0))
    
    total = sum(breakdown.values())
    weakest = min(breakdown, key=lambda k: breakdown[k])
    
    return {
        "total": round(total, 1),
        "breakdown": breakdown,
        "weakest": weakest,
        "llm_reason": llm.get("reason", "")
    }


# ── 变异算子 ──────────────────────────────────────────────

MUTATION_PROMPTS = {
    "D3": """当前 Skill 缺少论文引用。请：
1. 在 frontmatter 中识别出最相关的论文名称
2. 在 ## ① 算法原理 开头加入 > **论文**：{推测的论文名} | **arXiv**：{搜索结果}
3. 确保算法原理中提到论文的核心贡献
只输出修改后的完整 Skill markdown，不要解释。""",

    "D6": """当前 Skill 的 Python 代码无法运行或缺少 [✓] 测试输出。请：
1. 检查所有 import，确保使用标准库（numpy/pandas/sklearn/scipy）
2. 确保代码包含完整的示例数据（不依赖外部文件）
3. 在代码末尾加入 print("[✓] {算法名}测试通过")
4. 确保代码实际运行产生有意义的输出
只输出修改后的完整 Skill markdown，不要解释。""",

    "D7": """当前 Skill 的代码没有反映论文核心算法，只是 mock 数据或随机数据。请：
1. 从 ## ① 算法原理 提取核心数学公式
2. 将代码重写为真实实现（即使是简化版）
3. 代码中的变量名要对应论文中的符号（如 beta、gamma、theta）
4. 保留原有的业务案例背景
只输出修改后的完整 Skill markdown，不要解释。""",

    "D8": """当前 Skill 的算法原理缺乏数学直觉。请在 ## ① 算法原理 中：
1. 加入核心公式（使用 Python/LaTeX 风格，如 y = wx + b）
2. 用一句业务语言解释每个公式的含义
3. 加入"非共识迁移"段落：说明该算法原始来自哪个领域（如金融/流行病学），为何能降维打击跨境电商
只输出修改后的完整 Skill markdown，不要解释。""",

    "D9": """当前 Skill 的应用案例过于泛泛。请重写 ## ② 母婴出海应用案例：
1. 使用具体品类（如：婴儿暖奶器/婴儿推车/有机米粉）
2. 给出具体数字（如：库存 2000 件/日销 50 件/ROAS 3.2）
3. 预期产出要量化（如：库存周转率提升 28%/年化节省 45 万元）
4. 每个场景必须有三轨验证：成本验证/合规验证/风险验证
只输出修改后的完整 Skill markdown，不要解释。""",

    "D10": """当前 Skill 缺少三轨对抗验证。在每个应用场景的结尾，加入：
**三轨验证**：
1. **成本验证**：[执行该决策的显性+隐性成本]
2. **合规验证**：[是否触碰平台红线，如 Amazon 政策/GDPR]
3. **风险验证**：[商业侧次生风险，如引发竞品价格战]
只输出修改后的完整 Skill markdown，不要解释。""",

    "D11": """当前 Skill 缺乏跨学科迁移视角。在 ## ① 算法原理 末尾加入：
**非共识迁移**：该算法原本来自 [原始领域]（如流行病学/量子计算/博弈论）。
其"降维打击"在于：[跨境电商运营者通常会怎么做] → 而该算法通过 [核心机制] 反直觉地解决了 [具体问题]。
只输出修改后的完整 Skill markdown，不要解释。"""
}


def mutate_skill(skill_path: Path, weakest_dim: str, round_n: int) -> str:
    """根据最弱维度调用对应变异算子"""
    import requests, os
    
    content = skill_path.read_text(encoding="utf-8", errors="replace")
    prompt = MUTATION_PROMPTS.get(weakest_dim, MUTATION_PROMPTS["D9"])
    
    full_prompt = f"""以下是需要改进的 Skill 卡片（第 {round_n+1} 轮优化）：

{content}

改进指令：
{prompt}"""
    
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return content  # 无 API key 则返回原内容
    
    try:
        resp = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": full_prompt}],
                "max_tokens": 4000,
                "temperature": 0.3
            },
            timeout=60
        )
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"  变异失败: {e}")
        return content


# ── 主进化循环 ────────────────────────────────────────────

def evolve_skill(skill_id: str, dry_run: bool = False) -> dict:
    """
    对单个 Skill 运行达尔文进化循环
    返回：{"skill_id": ..., "initial_score": ..., "final_score": ..., "rounds": ...}
    """
    skill_path = find_skill_path(skill_id)
    if not skill_path:
        return {"error": f"Skill {skill_id} not found"}
    
    log_path = EVO_DIR / f"{skill_id}.jsonl"
    log_path.parent.mkdir(exist_ok=True)
    
    # 初始评分
    initial = evaluate_skill(skill_path)
    best_score = initial["total"]
    
    print(f"\n🧬 {skill_id}")
    print(f"   初始分: {best_score}/100  最弱: {initial['weakest']} ({initial['llm_reason'][:50]})")
    
    rounds_log = []
    
    for round_n in range(MAX_ROUNDS):
        if best_score >= TARGET_SCORE:
            print(f"   ✅ 达到目标分数 {TARGET_SCORE}，早停")
            break
        
        current = evaluate_skill(skill_path)
        
        # 变异
        new_content = mutate_skill(skill_path, current["weakest"], round_n)
        
        if dry_run:
            print(f"   [dry-run] Round {round_n+1}: would mutate {current['weakest']}")
            continue
        
        # 写入变异版本
        skill_path.write_text(new_content, encoding="utf-8")
        
        # 重新评分
        new_eval = evaluate_skill(skill_path)
        new_score = new_eval["total"]
        
        if new_score > best_score:
            best_score = new_score
            status = "keep"
            if not dry_run:
                subprocess.run(["git", "add", str(skill_path)], capture_output=True)
        else:
            # 回滚
            subprocess.run(["git", "checkout", "--", str(skill_path)], capture_output=True)
            status = "discard"
        
        round_log = {
            "round": round_n, "status": status,
            "score": new_score, "best": best_score,
            "mutated_dim": current["weakest"],
            "timestamp": datetime.now().isoformat()
        }
        rounds_log.append(round_log)
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(round_log, ensure_ascii=False) + "\n")
        
        icon = "✅" if status == "keep" else "↩️"
        print(f"   {icon} Round {round_n+1}: {status} {new_score:.1f}/100 (mutated: {current['weakest']})")
    
    final_score = best_score
    delta = final_score - initial["total"]
    
    print(f"   📊 最终: {initial['total']:.1f} → {final_score:.1f} (+{delta:.1f})")
    
    return {
        "skill_id": skill_id,
        "initial_score": initial["total"],
        "final_score": final_score,
        "delta": delta,
        "rounds": len(rounds_log),
        "rounds_log": rounds_log
    }


def find_skill_path(skill_id: str) -> Optional[Path]:
    for domain in VAULT.iterdir():
        if not domain.is_dir(): continue
        candidate = domain / f"{skill_id}.md"
        if candidate.exists():
            return candidate
    return None


# ── 批量进化（按领域分批）────────────────────────────────

def batch_evolve(
    domain: Optional[str] = None,
    priority: str = "no_paper",  # no_paper / no_code_algo / low_roi / all
    limit: int = 50
) -> list:
    """
    按领域分批进化
    priority: 优先进化哪类 Skill
      - no_paper: 无论文引用（270个，最严重）
      - no_code_algo: 代码无实质算法（37个）
      - low_roi: 无ROI量化（44个）
      - all: 全量按分数从低到高
    """
    import time
    
    # 筛选目标 Skill
    candidates = []
    
    search_root = VAULT / domain if domain else VAULT
    
    for d in sorted(search_root.iterdir() if domain else VAULT.iterdir()):
        if not d.is_dir() or (not domain and not d.name[0].isdigit()):
            continue
        for sf in d.iterdir():
            if not sf.name.startswith("Skill-") or not sf.name.endswith(".md"):
                continue
            content = sf.read_text(encoding="utf-8", errors="replace")
            
            if priority == "no_paper":
                import re
                if not re.search(r'arXiv|arxiv|NeurIPS|ICML|KDD|ICLR', content):
                    candidates.append(sf.stem)
            elif priority == "no_code_algo":
                blocks = re.findall(r'```python\n(.*?)```', content, re.DOTALL)
                algo_kws = ['def ', 'import ', 'sklearn', 'numpy', 'torch', 'scipy', 'for ', 'class ']
                if blocks and not any(kw in blocks[0] for kw in algo_kws):
                    candidates.append(sf.stem)
            elif priority == "all":
                candidates.append(sf.stem)
    
    target = candidates[:limit]
    print(f"\n🎯 批量进化: domain={domain or '全部'}, priority={priority}, 共 {len(target)} 个 Skill")
    
    results = []
    for i, skill_id in enumerate(target, 1):
        print(f"\n[{i}/{len(target)}]", end="")
        result = evolve_skill(skill_id)
        results.append(result)
        time.sleep(1)  # 避免 API 限速
    
    # 生成进化报告
    improved = [r for r in results if r.get("delta", 0) > 0]
    print(f"\n\n📊 批量进化完成")
    print(f"   进化 Skill 数: {len(results)}")
    print(f"   成功改进: {len(improved)}/{len(results)}")
    if improved:
        avg_delta = sum(r["delta"] for r in improved) / len(improved)
        print(f"   平均改进: +{avg_delta:.1f} 分")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Skill 达尔文进化引擎")
    parser.add_argument("--domain", help="指定域（如 01-因果推断）")
    parser.add_argument("--priority", default="no_paper",
                        choices=["no_paper", "no_code_algo", "low_roi", "all"])
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--skill", help="单个 Skill ID")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    if args.skill:
        evolve_skill(args.skill, dry_run=args.dry_run)
    else:
        batch_evolve(domain=args.domain, priority=args.priority, limit=args.limit)
```

---

## 五、按领域分批执行计划

| 批次 | 领域 | Skill数 | 优先级 | 预计改进 |
|------|------|--------|--------|---------|
| Batch 1 | 01-因果推断 | 30 | 无论文引用 | +8~15分/Skill |
| Batch 2 | 02-A_B实验 | 26 | 无论文引用 | +8~15分/Skill |
| Batch 3 | 03-时间序列 | 39 | 代码无算法 | +12~20分/Skill |
| Batch 4 | 04-供应链 | 129 | 全量（分3批）| +5~12分/Skill |
| Batch 5 | 06-增长模型 | 61 | 无ROI量化 | +8~12分/Skill |
| ... | 其余 20 个域 | 838 | 按质量分排序 | +5~10分/Skill |

---

## 六、进化日志格式

每个 Skill 一个 `.evo/{skill_id}.jsonl`，格式：

```json
{"round": 0, "status": "keep", "score": 72.5, "best": 72.5, "mutated_dim": "D3", "timestamp": "2026-07-04T10:00:00"}
{"round": 1, "status": "discard", "score": 69.0, "best": 72.5, "mutated_dim": "D7", "timestamp": "2026-07-04T10:05:00"}
{"round": 2, "status": "keep", "score": 84.3, "best": 84.3, "mutated_dim": "D9", "timestamp": "2026-07-04T10:10:00"}
```

---

## 七、与方案 B 的关系

| 维度 | 方案 A（达尔文）| 方案 B（用户反馈）|
|------|--------------|----------------|
| 触发来源 | 内部评分引擎（自驱）| 用户使用行为（外驱）|
| 运行频率 | 批次式（每月/每季）| 实时/每日 |
| 改进依据 | 9维度评分最弱项 | 用户点击/跳出/反馈 |
| 组合效果 | **叠加不冲突**：A 保底质量，B 贴近用户需求 |

**两套方案可以同时运行，进化日志分开记录，改进叠加积累。**
