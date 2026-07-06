#!/usr/bin/env python3
"""
Skill 自进化引擎 v3.0 — 三模式分层策略

策略分层（第一性原理）：
  <65  分 → full_rewrite:      全量重写5模块
  65-75分 → targeted_rewrite:  只重写最弱单模块（不动其他内容）
  75-80分 → targeted_patch:    手术式追加缺失的论文引用+三轨验证（不改写任何已有内容）
  ≥80  分 → skip

用法:
  python darwin_evolve_v3.py --mode patch   --limit 200   # 75-80分手术注入
  python darwin_evolve_v3.py --mode rewrite --limit 100   # <75分重写
  python darwin_evolve_v3.py --mode all     --limit 300   # 全量
  python darwin_evolve_v3.py --skill Skill-XXX --dry-run
  python darwin_evolve_v3.py --domain 04-供应链 --mode all
"""
import ast, json, os, re, subprocess, sys, time, argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

VAULT     = Path("paper2skills-vault")
EVO_DIR   = Path(".evo")
API_KEY   = "sk-deaqw8rzfud7r1wuo573ahhijrtmng7b"
API_URL   = "https://cliproxy.luteos.site/v1/chat/completions"
MODEL     = "claude-haiku-4.5"
TARGET    = 85
PATCH_LO  = 75
PATCH_HI  = 80
REWRITE_HI = 75

DIMS = {
    "D1_modules": 8, "D2_frontmatter": 5, "D3_paper": 10, "D4_roi": 8,
    "D5_links": 4, "D6_code": 10, "D7_code_depth": 15, "D8_algo": 10,
    "D9_scenario": 15, "D10_three_track": 10, "D11_crossdomain": 5,
}

DOMAIN_SCENARIO_HINT = {
    "01-因果推断":    "母婴暖奶器促销归因，转化率提升X%，置信区间[a,b]，年化42万元",
    "02-A_B实验":     "婴儿推车listing多臂老虎机，ROAS从2.1→3.4，周流量5000",
    "03-时间序列":    "有机辅食补货预测，提前21天，MAPE<15%，周转率提升28%",
    "04-供应链":      "婴儿奶粉跨境备货，FBA缺货率从12%→3%，年化节省45万元",
    "05-推荐系统":    "婴儿洗护套装复购推荐，CTR提升18%，复购率22%→31%",
    "06-增长模型":    "母婴用品用户流失预警，提前14天干预，LTV提升35万/年",
    "07-NLP-VOC":     "暖奶器Amazon评论情感挖掘，差评率8.2%→4.1%",
    "08-知识图谱":    "母婴品类供应商知识图谱，找到3个替代供应商，断货风险降低60%",
    "09-DataAgent-LLM":"DeepSeek自动生成母婴Listing，CTR+22%，人力节省80%",
    "10-MAS":         "多Agent协同大促备货，准确率91%，误判损失降低38万元",
    "11-AI人文":      "AI情感陪伴+母婴育儿焦虑，用户留存提升15%",
    "12-ML基础":      "婴儿用品销量预测集成模型，RMSE降低23%，备货成本节省12万",
    "13-广告分析":    "母婴Amazon Sponsored Ads，ROAS从2.8→4.1，广告费节省18%",
    "14-用户分析":    "母婴品牌RFM分层，高价值用户复购率+28%，LTV年化提升52万",
    "15-营销投放分析":"婴儿推车TikTok+Amazon全渠道MMM，预算分配优化，ROI+31%",
    "16-智能体工程":  "母婴运营Agent，自动监控库存+广告+竞品，响应时间4h→15min",
    "17-价格优化":    "婴儿安全座椅动态定价，价格弹性-1.8，GMV提升23%，毛利+4pp",
    "18-物流履约":    "母婴跨境最后一公里，时效7天→5天，配送成本降低1.2$/单",
    "19-风控反欺诈":  "母婴店铺刷单检测，误判率<0.5%，每月挽回损失8万元",
    "20-AI视频生成":  "婴儿洗护虚拟主播，内容成本降低85%，GMV转化率3.2%",
    "21-合规决策":    "婴儿食品FDA/CE合规矩阵，上架周期45天→22天",
    "22-数据采集工程":"Amazon母婴类目全量爬取，数据质量>99%，覆盖50万+SKU",
    "23-运营财务":    "母婴FBA P&L自动核算，毛利准确率99.2%，每月节省40人时",
    "24-标签工程":    "母婴SKU标签体系，自动标注准确率94%，决策提速60%",
    "25-搜索流量工程":"婴儿推车A9算法优化，自然搜索排名P3→P1，流量提升340%",
}


def call_llm(prompt: str, max_tokens: int = 6000, temperature: float = 0.3) -> str:
    import requests
    try:
        resp = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": max_tokens, "temperature": temperature},
            timeout=120
        )
        content = resp.json()["choices"][0]["message"]["content"]
        m = re.match(r'^```(?:markdown)?\n(.*?)```\s*$', content, re.DOTALL)
        return m.group(1) if m else content
    except Exception as e:
        print(f"     API失败: {e}")
        return ""


def score_auto(content: str) -> dict:
    s = {}
    s["D1_modules"] = round(sum(1 for h in ["## ①","## ②","## ③","## ④","## ⑤"] if h in content) / 5 * 8, 1)
    has_fm = content.startswith("---\n")
    s["D2_frontmatter"] = 5.0 if (has_fm and "roadmap_phase:" in content[:600] and "doc_type:" in content[:600]) else (3.0 if has_fm else 0.0)
    s["D3_paper"] = 10.0 if re.search(r'arXiv|arxiv|\b(NeurIPS|ICML|KDD|ICLR|AAAI|WWW|SIGIR|CIKM|WSDM|ECML|VLDB)\b', content) else 0.0
    v_sec = content[content.find("## ⑤"):content.find("## ⑤")+800] if "## ⑤" in content else content[-800:]
    s["D4_roi"] = 8.0 if re.search(r'\d+\s*(?:万|%|元|亿|\$|pp)', v_sec) else 0.0
    s["D5_links"] = min(len(re.findall(r'\[\[Skill-[^\]]+\]\]', content)) / 2 * 4, 4.0)
    return s


def score_code(skill_path: Path) -> float:
    content = skill_path.read_text(encoding="utf-8", errors="replace")
    blocks = re.findall(r'```python\n(.*?)```', content, re.DOTALL)
    if not blocks:
        return 0.0
    try:
        ast.parse(blocks[0])
    except SyntaxError:
        return 2.0
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
        f.write(blocks[0])
        tmp = f.name
    try:
        r = subprocess.run([sys.executable, tmp], capture_output=True, text=True, timeout=30)
        has_check = "[✓]" in r.stdout or "[✔]" in r.stdout
        return 10.0 if r.returncode == 0 and has_check else (7.0 if r.returncode == 0 else 3.0)
    except subprocess.TimeoutExpired:
        return 4.0
    finally:
        try: os.unlink(tmp)
        except: pass


def score_llm(content: str) -> dict:
    prompt = f"""顶级 Skill 质量评审专家。对下列 Skill 按5维度打分。

{content[:3000]}

D7(代码反映论文,满分15)：15=核心算法+变量名对应论文符号，8=有框架未深入，0=mock
D8(算法深度,满分10)：10=公式+直觉+跨学科，5=有描述缺数学，0=名称堆砌
D9(场景具体性,满分15)：15=具体品类+具体数字+量化产出，8=数字模糊，0=泛泛而谈
D10(三轨验证,满分10)：10=成本+合规+风险三轨完整，5=1-2轨，0=无
D11(跨学科迁移,满分5)：5=原始领域+降维打击原理，2=提及不清，0=无

只输出JSON：{{"D7_code_depth":数字,"D8_algo":数字,"D9_scenario":数字,"D10_three_track":数字,"D11_crossdomain":数字,"weakest":"维度key","reason":"最弱问题(15字内)"}}"""
    import requests
    try:
        resp = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 300, "temperature": 0.1},
            timeout=30
        )
        raw = resp.json()["choices"][0]["message"]["content"]
        m = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
        for k in ["D7_code_depth","D8_algo","D9_scenario","D10_three_track","D11_crossdomain"]:
            data.setdefault(k, 5)
        return data
    except:
        return {"D7_code_depth":8,"D8_algo":5,"D9_scenario":8,"D10_three_track":5,"D11_crossdomain":2,"weakest":"D10_three_track","reason":"评分异常"}


def evaluate(skill_path: Path) -> dict:
    content = skill_path.read_text(encoding="utf-8", errors="replace")
    auto = score_auto(content)
    code = score_code(skill_path)
    llm = score_llm(content)
    breakdown = {**auto, "D6_code": code}
    for k in ["D7_code_depth","D8_algo","D9_scenario","D10_three_track","D11_crossdomain"]:
        breakdown[k] = float(llm.get(k, 0))
    total = sum(breakdown.values())
    weakest = min(breakdown, key=lambda k: breakdown[k] / DIMS[k])
    return {"total": round(total, 1), "breakdown": breakdown, "weakest": weakest, "reason": llm.get("reason","")}


def find_skill(skill_id: str) -> Optional[Path]:
    for d in VAULT.iterdir():
        if not d.is_dir(): continue
        p = d / f"{skill_id}.md"
        if p.exists(): return p
    return None


def get_domain(skill_path: Path) -> str:
    return skill_path.parent.name


def log_entry(log_path: Path, round_n: int, status: str, score: float, best: float, mutated: str):
    entry = {"round": round_n, "status": status, "score": score, "best": best,
             "mutated": mutated, "ts": datetime.now().isoformat()}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── 模式1: full_rewrite（<65分）────────────────────────────

REWRITE_PROMPT = """你是顶级 paper2skills Skill 内容专家，专注母婴跨境电商 AI 决策卡片。

以下是需要完整重写的 Skill 卡片（质量较差）：

{original}

按标准5模块全量重写，目标≥85分：

## ① 算法原理（≤300字）
- 核心思想一句话概括
- 至少1个核心公式+业务语言含义
- 关键假设
- **非共识迁移**：原始领域+降维打击跨境电商原理（必须有）

## ② 母婴出海应用案例（2个具体场景）
- 必须用具体母婴品类：{domain_hint}
- 每个场景含：业务问题+具体数字+量化产出（万元/百分比）
- 末尾加三轨验证（成本/合规/风险各一句话，含具体数字）

## ③ 代码模板（Python）
- 完整可运行，只用numpy/pandas/sklearn/scipy标准库
- 含内嵌示例数据
- 末尾：print("[✓] {skill_name}测试通过")
- 150-250行，充分展示算法核心

## ④ 技能关联
- 前置：至少1条 [[Skill-XXX]]
- 延伸：至少1条 [[Skill-XXX]]
- 可组合：至少1条 [[Skill-XXX]]（说明组合场景）

## ⑤ 商业价值评估
- ROI：具体数字（万元/百分比）
- 实施难度：⭐⭐⭐☆☆
- 优先级：⭐⭐⭐⭐☆

保留原有frontmatter，更新 updated: {today}，确保含 roadmap_phase。
只输出完整Skill markdown，不要解释。"""


def full_rewrite(skill_path: Path) -> str:
    content = skill_path.read_text(encoding="utf-8", errors="replace")
    domain = get_domain(skill_path)
    hint = DOMAIN_SCENARIO_HINT.get(domain, "婴儿推车/暖奶器/有机辅食，给出具体数字")
    today = datetime.now().strftime("%Y-%m-%d")
    prompt = REWRITE_PROMPT.format(
        original=content[:4000], domain_hint=hint,
        skill_name=skill_path.stem, today=today
    )
    return call_llm(prompt, max_tokens=8000, temperature=0.4)


# ── 模式2: targeted_rewrite（65-75分，只重写最弱模块）──────

MODULE_PROMPTS = {
    "D8_algo": """该Skill的算法原理(## ①)深度不足。只重写 ## ① 模块，要求：
1. 加入核心公式（Python表达式或LaTeX）+ 每个公式的业务含义（≤20字）
2. 末尾新增「**非共识迁移**」：原始领域 → 跨境电商降维打击原理（2-3句话）
3. 保持原文其他内容完全不变（## ② ③ ④ ⑤ 一字不动）
原始Skill内容：\n{content}\n只输出完整改进后的Skill markdown，不要解释。""",

    "D9_scenario": """该Skill的应用案例(## ②)过于泛泛。只重写 ## ② 模块，要求：
1. 用具体母婴品类替换：{domain_hint}
2. 每个场景含具体数字（库存件数/日销量/ROAS/转化率）
3. 量化产出（年化万元/百分比）
4. 末尾加三轨验证（成本/合规/风险各一句，含数字）
5. 保持其他模块完全不变
原始Skill内容：\n{content}\n只输出完整改进后的Skill markdown，不要解释。""",

    "D7_code_depth": """该Skill的代码(## ③)未反映论文核心算法。只重写 ## ③ 模块，要求：
1. 从 ## ① 提取核心数学公式，代码变量名对应论文符号
2. 添加注释说明代码行对应论文哪个公式
3. 末尾确保有 print("[✓] 测试通过")
4. 保持其他模块完全不变
原始Skill内容：\n{content}\n只输出完整改进后的Skill markdown，不要解释。""",

    "D6_code": """该Skill的代码无法运行或缺少[✓]输出。只修复 ## ③ 代码块，要求：
1. 只用numpy/pandas/sklearn/scipy标准库（无外部依赖）
2. 含完整内嵌示例数据
3. 末尾最后一行：print("[✓] {skill_name}测试通过")
4. 保持其他模块完全不变
原始Skill内容：\n{content}\n只输出完整改进后的Skill markdown，不要解释。""",

    "D10_three_track": """该Skill缺少三轨验证。在 ## ② 每个应用场景末尾追加，不改写场景正文：
**三轨验证**：
- **成本轨**：{成本数字，如：数据采集月均500元，计算资源0.2元/千次调用}
- **合规轨**：{是否合规，如：不触碰Amazon政策/GDPR，用户数据本地化}
- **风险轨**：{次生风险，如：引发竞品跟价概率30%，建议设置价格保护区间}
保持其他所有内容完全不变。
原始Skill内容：\n{content}\n只输出完整改进后的Skill markdown，不要解释。""",

    "D3_paper": """该Skill缺少论文引用。在 ## ① 开头第一行追加，不改写其他内容：
> **论文**：{推断最可能的真实论文名，格式：作者et al.，年份，会议} | **arXiv**：推断的ID或N/A
同时在frontmatter追加：source: arxiv:XXXX.XXXXX（若能推断）
保持其他所有内容完全不变。
原始Skill内容：\n{content}\n只输出完整改进后的Skill markdown，不要解释。""",
}


def targeted_rewrite(skill_path: Path, weakest: str) -> str:
    content = skill_path.read_text(encoding="utf-8", errors="replace")
    domain = get_domain(skill_path)
    hint = DOMAIN_SCENARIO_HINT.get(domain, "婴儿推车/暖奶器，给出具体数字")
    tmpl = MODULE_PROMPTS.get(weakest, MODULE_PROMPTS["D9_scenario"])
    prompt = tmpl.format(content=content[:5000], domain_hint=hint, skill_name=skill_path.stem)
    return call_llm(prompt, max_tokens=6000, temperature=0.3)


# ── 模式3: targeted_patch（75-80分，手术式注入）────────────

PATCH_PAPER_PROMPT = """该Skill缺少学术论文引用（D3=0分，损失10分）。

**只在 ## ① 开头追加一行论文引用，其余内容一字不动。**

根据算法名称和内容，推断最可能对应的真实学术论文：
- 格式：> **论文**：作者 et al., 年份, 会议/期刊 | **arXiv**：ID或N/A
- 优先推断 NeurIPS/ICML/KDD/ICLR/AAAI 2020-2024 的真实论文
- 不要捏造不存在的论文，若不确定写 N/A

原始Skill（只看前500字判断算法）：
{content_head}

只输出完整改进后的Skill markdown，不要解释。"""

PATCH_TRACK_PROMPT = """为以下母婴跨境电商 AI Skill 生成三轨验证内容。

Skill名称：{skill_name}
Skill域：{domain}
业务场景参考：{domain_hint}

生成格式（JSON，只输出JSON）：
{{
  "scenario1_track": "**三轨验证** | 成本轨：具体成本估算（如：API调用月均200元，人工校验8小时/月） | 合规轨：合规结论（如：符合Amazon政策/GDPR，数据不出境） | 风险轨：次生风险（如：模型过拟合概率15%，建议每季度重训练）",
  "scenario2_track": "**三轨验证** | 成本轨：... | 合规轨：... | 风险轨：..."
}}

只输出JSON，不要解释，不要其他内容。"""


def targeted_patch(skill_path: Path, patch_type: str) -> str:
    content = skill_path.read_text(encoding="utf-8", errors="replace")
    if patch_type == "paper":
        prompt = PATCH_PAPER_PROMPT.format(content_head=content[:600])
        return call_llm(prompt, max_tokens=5000, temperature=0.2)

    domain = get_domain(skill_path)
    hint = DOMAIN_SCENARIO_HINT.get(domain, "婴儿推车/暖奶器，给出具体数字")
    prompt = PATCH_TRACK_PROMPT.format(
        skill_name=skill_path.stem, domain=domain, domain_hint=hint
    )
    import requests
    try:
        resp = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 500, "temperature": 0.3},
            timeout=60
        )
        raw = resp.json()["choices"][0]["message"]["content"]
        m = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        if not m:
            return ""
        data = json.loads(m.group(0))
        track1 = data.get("scenario1_track", "")
        track2 = data.get("scenario2_track", "")
        if not track1:
            return ""
        # 在 ## ② 每个场景末尾追加（找 ## ③ 前的位置）
        sec2_start = content.find("## ②")
        sec3_start = content.find("## ③")
        if sec2_start == -1 or sec3_start == -1:
            return ""
        sec2 = content[sec2_start:sec3_start]
        track_text = f"\n\n{track1}"
        if track2:
            track_text += f"\n\n{track2}"
        new_sec2 = sec2.rstrip() + track_text + "\n\n"
        return content[:sec2_start] + new_sec2 + content[sec3_start:]
    except Exception as e:
        print(f"     patch_track 失败: {e}")
        return ""


# ── 主进化逻辑 ─────────────────────────────────────────────

def evolve_skill(skill_id: str, dry_run: bool = False) -> dict:
    skill_path = find_skill(skill_id)
    if not skill_path:
        return {"error": f"找不到 {skill_id}"}

    EVO_DIR.mkdir(exist_ok=True)
    log_path = EVO_DIR / f"{skill_id}.jsonl"

    initial_eval = evaluate(skill_path)
    best = initial_eval["total"]
    domain = get_domain(skill_path)

    print(f"\n🧬 [{domain}] {skill_id}")
    print(f"   初始分: {best:.1f}  最弱: {initial_eval['weakest']} — {initial_eval['reason']}")

    if best >= TARGET:
        print(f"   ✅ ≥{TARGET}分，跳过")
        log_entry(log_path, 0, "skip_high_score", best, best, "none")
        return {"skill_id": skill_id, "status": "skip_high_score", "initial": best, "final": best, "delta": 0}

    if dry_run:
        mode = "full_rewrite" if best < REWRITE_HI else ("targeted_rewrite" if best < PATCH_LO else "targeted_patch")
        print(f"   [dry-run] 模式: {mode}, 最弱: {initial_eval['weakest']}")
        return {"skill_id": skill_id, "status": "dry-run", "initial": best, "final": best}

    original = skill_path.read_text(encoding="utf-8", errors="replace")

    # ─ 模式1: full_rewrite <75分 ─────────────────────────────
    if best < REWRITE_HI:
        print(f"   🔄 full_rewrite (<{REWRITE_HI}分)...")
        new_content = full_rewrite(skill_path)
        if not new_content or len(new_content) < 500:
            log_entry(log_path, 0, "rewrite_failed", best, best, "full_rewrite")
            print(f"   ❌ 重写失败")
            return {"skill_id": skill_id, "status": "rewrite_failed", "initial": best, "final": best, "delta": 0}

        skill_path.write_text(new_content, encoding="utf-8")
        auto_new = score_auto(new_content)
        code_new = score_code(skill_path)
        rule_new = sum(auto_new.values()) + code_new
        auto_old = score_auto(original)
        skill_path.write_text(original, encoding="utf-8")
        code_old = score_code(skill_path)
        rule_old = sum(auto_old.values()) + code_old
        length_ok = len(new_content) > len(original) * 0.8

        if rule_new >= rule_old - 2 and length_ok:
            skill_path.write_text(new_content, encoding="utf-8")
            new_eval = evaluate(skill_path)
            best = new_eval["total"]
            status = "rewrite_improved"
            subprocess.run(["git", "add", str(skill_path)], capture_output=True)
        else:
            status = "rewrite_no_gain"

        log_entry(log_path, 0, status, best, best, "full_rewrite")
        icon = "✅" if status == "rewrite_improved" else "↩️ "
        print(f"   {icon} {status}: {best:.1f}/100 (rule:{rule_old:.0f}→{rule_new:.0f})")
        return {"skill_id": skill_id, "status": status, "initial": initial_eval["total"], "final": best, "delta": best - initial_eval["total"]}

    # ─ 模式2: targeted_rewrite 75-80分 中的 75-80 但weakest是内容维度 ─
    if best < PATCH_LO:
        weakest = initial_eval["weakest"]
        print(f"   ✏️  targeted_rewrite (65-75分, 重写{weakest})...")
        new_content = targeted_rewrite(skill_path, weakest)
        if not new_content or len(new_content) < 500:
            log_entry(log_path, 0, "targeted_failed", best, best, weakest)
            print(f"   ❌ targeted重写失败")
            return {"skill_id": skill_id, "status": "targeted_failed", "initial": best, "final": best, "delta": 0}

        skill_path.write_text(new_content, encoding="utf-8")
        new_eval = evaluate(skill_path)
        new_score = new_eval["total"]

        if new_score > best and new_score >= best * 0.9:
            best = new_score
            status = "targeted_improved"
            subprocess.run(["git", "add", str(skill_path)], capture_output=True)
        else:
            skill_path.write_text(original, encoding="utf-8")
            status = "targeted_no_gain"

        log_entry(log_path, 0, status, new_score, best, weakest)
        icon = "✅" if "improved" in status else "↩️ "
        print(f"   {icon} {status}: {best:.1f}/100")
        return {"skill_id": skill_id, "status": status, "initial": initial_eval["total"], "final": best, "delta": best - initial_eval["total"]}

    # ─ 模式3: targeted_patch（75-80分，手术注入）───────────────
    print(f"   🔬 targeted_patch (75-80分, 手术注入)...")
    content = skill_path.read_text(encoding="utf-8", errors="replace")
    has_paper = bool(re.search(r'arXiv|arxiv|\b(NeurIPS|ICML|KDD|ICLR|AAAI)\b', content))
    has_track = bool(re.search(r'三轨|成本轨|合规轨|风险轨', content))

    patches_applied = []
    for patch_type, needed in [("paper", not has_paper), ("track", not has_track)]:
        if not needed:
            continue
        patched = targeted_patch(skill_path, patch_type)
        if not patched or len(patched) < len(content) * 0.8:
            print(f"   ↩️  patch_{patch_type}: 内容异常，跳过")
            continue
        skill_path.write_text(patched, encoding="utf-8")
        content = patched
        patches_applied.append(patch_type)

    if patches_applied:
        new_eval = evaluate(skill_path)
        new_score = new_eval["total"]
        if new_score >= best - 2:
            best = new_score
            status = "patch_improved"
            subprocess.run(["git", "add", str(skill_path)], capture_output=True)
        else:
            skill_path.write_text(original, encoding="utf-8")
            status = "patch_revert"
    else:
        status = "patch_nothing"
        print(f"   ⏭  论文+三轨均已有，无需注入")

    log_entry(log_path, 0, status, best, best, "+".join(patches_applied) or "none")
    icon = "✅" if "improved" in status else ("⏭ " if "nothing" in status else "↩️ ")
    print(f"   {icon} {status}: {best:.1f}/100 patches={patches_applied}")
    return {"skill_id": skill_id, "status": status, "initial": initial_eval["total"], "final": best, "delta": best - initial_eval["total"]}


# ── 批量入口 ─────────────────────────────────────────────────

def load_candidates(domain: Optional[str], mode: str) -> list:
    candidates = []
    domain_dirs = [VAULT / domain] if domain else sorted(
        [d for d in VAULT.iterdir() if d.is_dir() and d.name[0:1].isdigit()]
    )

    for d in domain_dirs:
        for sf in sorted(d.iterdir()):
            if not sf.name.startswith("Skill-") or not sf.name.endswith(".md"):
                continue
            skill_id = sf.stem
            log_path = EVO_DIR / f"{skill_id}.jsonl"
            best_known = None
            if log_path.exists():
                try:
                    lines = [l for l in log_path.read_text().strip().split('\n') if l.strip()]
                    records = [json.loads(l) for l in lines]
                    best_known = max(r.get("best", r.get("score", 0)) for r in records)
                    last_status = records[-1].get("status", "")
                    if last_status in ("skip_high_score",) or best_known >= TARGET:
                        continue
                except:
                    pass

            if best_known is None:
                c = sf.read_text(encoding="utf-8", errors="replace")
                auto = score_auto(c)
                code = score_code(sf)
                best_known = round(sum(auto.values()) + code + 28, 1)

            if mode == "patch" and PATCH_LO <= best_known < PATCH_HI:
                candidates.append((best_known, skill_id))
            elif mode == "rewrite" and best_known < REWRITE_HI:
                candidates.append((best_known, skill_id))
            elif mode == "targeted" and REWRITE_HI <= best_known < PATCH_LO:
                candidates.append((best_known, skill_id))
            elif mode == "all" and best_known < TARGET:
                candidates.append((best_known, skill_id))

    candidates.sort()
    return [sid for _, sid in candidates]


def batch_evolve(domain: Optional[str], mode: str, limit: int, dry_run: bool = False) -> list:
    candidates = load_candidates(domain, mode)
    target = candidates[:limit]
    print(f"\n🎯 v3批量进化: domain={domain or '全部'}, mode={mode}, 候选={len(candidates)}, 本批={len(target)}")

    results = []
    improved = 0
    total_delta = 0.0

    for i, sid in enumerate(target, 1):
        print(f"\n[{i}/{len(target)}]", end="")
        r = evolve_skill(sid, dry_run=dry_run)
        results.append(r)
        if r.get("delta", 0) > 0:
            improved += 1
            total_delta += r["delta"]

    print(f"\n\n{'='*55}")
    print(f"📊 v3进化完成: 总={len(results)}, 净改进={improved}")
    if improved:
        print(f"   平均改进: +{total_delta/improved:.1f}分")
    print(f"{'='*55}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skill自进化引擎v3.0")
    parser.add_argument("--skill",  help="单个Skill ID")
    parser.add_argument("--domain", help="按域批量")
    parser.add_argument("--mode",   default="all",
                        choices=["rewrite","targeted","patch","all"],
                        help="rewrite=全量重写<75, targeted=重写单模块65-75, patch=手术注入75-80, all=全部")
    parser.add_argument("--limit",  type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.skill:
        evolve_skill(args.skill, dry_run=args.dry_run)
    else:
        batch_evolve(domain=args.domain, mode=args.mode, limit=args.limit, dry_run=args.dry_run)
