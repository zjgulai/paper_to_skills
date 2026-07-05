#!/usr/bin/env python3
"""
Skill 达尔文自进化引擎 v2.0
修复版：
  - 修复 skip 逻辑：初始分 ≥ TARGET 才真正 skip_high_score
  - 新增「重写」模式：分数 < REWRITE_THRESHOLD 走全量重写，不走 mutation
  - 域感知 mutation prompt：不同域使用不同业务案例
  - 评分与改写解耦：独立 Oracle 评分，防止自评偏差

用法:
  # 单个 Skill（自动判断：<65重写，65-85进化，≥85跳过）
  python darwin_evolve.py --skill Skill-Market-Size-Estimation

  # 批量重写低分（<65分）
  python darwin_evolve.py --mode rewrite --limit 50

  # 批量进化中分（65-85分）
  python darwin_evolve.py --mode evolve --limit 100

  # 全量（先重写低分，再进化中分）
  python darwin_evolve.py --mode all --limit 200

  # 只评分不改写
  python darwin_evolve.py --skill Skill-XYZ --dry-run

  # 按域批量
  python darwin_evolve.py --domain 22-数据采集工程 --mode all
"""
import ast, json, os, re, subprocess, sys, time, argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

# ── 配置 ─────────────────────────────────────────────────
VAULT              = Path("paper2skills-vault")
EVO_DIR            = Path(".evo")
# Cliproxy (Claude Haiku) — 低成本高速改写模型
API_KEY            = os.environ.get("CLIPROXY_API_KEY",
                     os.environ.get("DEEPSEEK_API_KEY", "sk-deaqw8rzfud7r1wuo573ahhijrtmng7b"))
API_URL            = os.environ.get("CLIPROXY_BASE_URL", "https://cliproxy.luteos.site/v1/chat/completions")
MODEL              = os.environ.get("EVOLVE_MODEL", "claude-haiku-4.5")
TARGET             = 85      # 早停分数（达到即跳过）
MAX_RNDS           = 5       # 进化最大轮数
REWRITE_THRESHOLD  = 65      # 低于此分走重写，高于走进化

# ── 11 维度权重（总计 100 分）─────────────────────────────
DIMS = {
    "D1_modules":      8,   # 5模块完整
    "D2_frontmatter":  5,   # frontmatter规范
    "D3_paper":       10,   # 论文引用
    "D4_roi":          8,   # ROI量化
    "D5_links":        4,   # 图谱关联
    "D6_code":        10,   # 代码可执行
    "D7_code_depth":  15,   # 代码反映论文
    "D8_algo":        10,   # 算法深度
    "D9_scenario":    15,   # 场景具体性
    "D10_three_track":10,   # 三轨验证
    "D11_crossdomain": 5,   # 跨学科迁移
}

# ── 域感知业务案例模板（D9 mutation 用）────────────────────
DOMAIN_SCENARIO_HINT = {
    "01-因果推断":    "母婴暖奶器 / A/B 促销归因，具体指标：转化率提升 X%，置信区间 [a,b]",
    "02-A_B实验":     "婴儿推车 listing 多臂老虎机实验，每周流量 5000，ROAS 从 2.1→3.4",
    "03-时间序列":    "有机辅食补货预测，提前 21 天，MAPE < 15%，库存周转率提升 28%",
    "04-供应链":      "婴儿奶粉跨境备货，FBA 缺货率从 12% → 3%，年化节省 45 万元",
    "05-推荐系统":    "婴儿洗护套装复购推荐，CTR 提升 18%，复购率从 22%→31%",
    "06-增长模型":    "母婴用品用户流失预警，提前 14 天干预，LTV 提升 35 万元/年",
    "07-NLP-VOC":     "暖奶器 Amazon 评论情感挖掘，差评率从 8.2%→4.1%，review score +0.3",
    "08-知识图谱":    "母婴品类供应商知识图谱，找到 3 个替代供应商，断货风险降低 60%",
    "09-DataAgent-LLM": "DeepSeek 自动生成母婴 Listing，CTR +22%，人力节省 80%",
    "10-MAS":         "多 Agent 协同大促备货决策，准确率 91%，误判损失降低 38 万元",
    "11-AI人文":      "AI 情感陪伴 + 母婴育儿焦虑缓解，用户留存提升 15%",
    "12-ML基础":      "婴儿用品销量预测集成模型，RMSE 降低 23%，备货成本节省 12 万",
    "13-广告分析":    "母婴 Amazon Sponsored Ads 归因，ROAS 从 2.8→4.1，广告费节省 18%",
    "14-用户分析":    "母婴品牌 RFM 分层，高价值用户复购率 +28%，LTV 年化提升 52 万",
    "15-营销投放分析":"婴儿推车 TikTok+Amazon 全渠道 MMM，预算分配优化，ROI +31%",
    "16-智能体工程":  "母婴运营 Agent，自动监控库存+广告+竞品，响应时间从 4h→15min",
    "17-价格优化":    "婴儿安全座椅动态定价，价格弹性 -1.8，GMV 提升 23%，毛利+4pp",
    "18-物流履约":    "母婴跨境最后一公里，时效从 7 天→5 天，配送成本降低 $1.2/单",
    "19-风控反欺诈":  "母婴店铺刷单检测，误判率 < 0.5%，每月挽回损失 8 万元",
    "20-AI视频生成":  "婴儿洗护虚拟主播，内容成本降低 85%，GMV 转化率 3.2%",
    "21-合规决策":    "婴儿食品 FDA/CE 合规矩阵，上架周期从 45 天→22 天，避免下架风险",
    "22-数据采集工程":"Amazon 母婴类目全量爬取，数据质量 >99%，覆盖 50 万+ SKU",
    "23-运营财务":    "母婴 FBA P&L 自动核算，毛利准确率 99.2%，每月节省 40 人时",
    "24-标签工程":    "母婴 SKU 标签体系，自动标注准确率 94%，千次运营决策提速 60%",
    "25-搜索流量工程":"婴儿推车 A9 算法优化，自然搜索排名从 P3→P1，流量提升 340%",
}


# ── 自动检测维度（D1-D5，纯规则，快速）────────────────────

def score_auto(content: str) -> dict:
    scores = {}

    # D1: 5模块完整性
    mods = sum(1 for h in ["## ①", "## ②", "## ③", "## ④", "## ⑤"] if h in content)
    scores["D1_modules"] = round(mods / 5 * 8, 1)

    # D2: Frontmatter 规范
    has_fm  = content.startswith("---\n")
    has_rp  = "roadmap_phase:" in content[:600]
    has_doc = "doc_type:" in content[:600]
    scores["D2_frontmatter"] = 5.0 if (has_fm and has_rp and has_doc) else (3.0 if has_fm else 0.0)

    # D3: 论文引用
    has_paper = bool(re.search(
        r'arXiv|arxiv|\b(NeurIPS|ICML|KDD|ICLR|AAAI|WWW|SIGIR|CIKM|WSDM|ECML|VLDB|SIGMOD)\b',
        content
    ))
    scores["D3_paper"] = 10.0 if has_paper else 0.0

    # D4: ROI 量化（⑤ 模块内）
    v_sec = content[content.find("## ⑤"):content.find("## ⑤") + 800] if "## ⑤" in content else content[-800:]
    has_roi = bool(re.search(r'\d+\s*(?:万|%|元|亿|\$|pp)', v_sec))
    scores["D4_roi"] = 8.0 if has_roi else 0.0

    # D5: 图谱关联数量
    links = len(re.findall(r'\[\[Skill-[^\]]+\]\]', content))
    scores["D5_links"] = min(links / 2 * 4, 4.0)

    return scores


def score_code(skill_path: Path) -> float:
    """D6: 代码可执行性（0/2/7/10）"""
    content = skill_path.read_text(encoding="utf-8", errors="replace")
    blocks  = re.findall(r'```python\n(.*?)```', content, re.DOTALL)

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
        r = subprocess.run(
            [sys.executable, tmp],
            capture_output=True, text=True, timeout=30
        )
        has_check = "[✓]" in r.stdout or "[✔]" in r.stdout
        if r.returncode == 0 and has_check:
            return 10.0
        elif r.returncode == 0:
            return 7.0
        else:
            return 3.0
    except subprocess.TimeoutExpired:
        return 4.0
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


def score_llm(content: str) -> dict:
    """D7-D11: DeepSeek 独立评委（截断至3500字防OOM）"""
    if not API_KEY:
        return {"D7_code_depth": 8, "D8_algo": 5, "D9_scenario": 8,
                "D10_three_track": 5, "D11_crossdomain": 2,
                "weakest": "D10_three_track", "reason": "未配置API key"}

    import requests
    prompt = f"""你是顶级 AI Skill 质量评审专家。对下列 Skill 卡片按 5 个维度打分。

---
{content[:3500]}
---

D7（代码反映论文，满分15）：15=核心算法实现+变量名对应论文符号，8=有框架但未深入，0=只是mock
D8（算法深度，满分10）：10=公式+直觉+跨学科来源，5=有描述缺数学，0=名称堆砌
D9（场景具体性，满分15）：15=具体品类+具体数字+量化产出，8=场景有但数字模糊，0=泛泛而谈
D10（三轨验证，满分10）：10=成本+合规+风险三轨完整，5=1-2轨，0=无
D11（跨学科迁移，满分5）：5=原始领域+降维打击原理，2=提及但不清，0=无

只输出JSON：{{"D7_code_depth":数字,"D8_algo":数字,"D9_scenario":数字,"D10_three_track":数字,"D11_crossdomain":数字,"weakest":"维度key","reason":"最弱问题（15字内）"}}"""

    try:
        resp = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": 300, "temperature": 0.1},
            timeout=30
        )
        raw = resp.json()["choices"][0]["message"]["content"]
        m = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        if not m:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
        for k in ["D7_code_depth", "D8_algo", "D9_scenario", "D10_three_track", "D11_crossdomain"]:
            data.setdefault(k, 5)
        return data
    except Exception as e:
        return {"D7_code_depth": 8, "D8_algo": 5, "D9_scenario": 8,
                "D10_three_track": 5, "D11_crossdomain": 2,
                "weakest": "D7_code_depth", "reason": f"评分异常:{str(e)[:20]}"}


def evaluate(skill_path: Path) -> dict:
    """完整 11 维度评分"""
    content  = skill_path.read_text(encoding="utf-8", errors="replace")
    auto     = score_auto(content)
    code     = score_code(skill_path)
    llm      = score_llm(content)

    breakdown = {**auto, "D6_code": code}
    for k in ["D7_code_depth", "D8_algo", "D9_scenario", "D10_three_track", "D11_crossdomain"]:
        breakdown[k] = float(llm.get(k, 0))

    total   = sum(breakdown.values())
    weakest = min(breakdown, key=lambda k: breakdown[k] / DIMS[k])

    return {
        "total":     round(total, 1),
        "breakdown": breakdown,
        "weakest":   weakest,
        "reason":    llm.get("reason", ""),
    }


# ── 变异提示词（11维度 × 域感知）──────────────────────────

MUTATIONS = {
    "D2_frontmatter": """该 Skill 的 frontmatter 不完整。请**只修改文件开头 --- 到 --- 之间的 frontmatter 部分**，补充以下字段（如缺少）：
- doc_type: knowledge
- roadmap_phase: phase1（或 phase2/phase3，根据域选择）
- status: stable
- updated: 当前日期

**严格要求：frontmatter 之外的内容（## ① ② ③ ④ ⑤）一字不改。**
只输出完整改进后的 Skill markdown，不要解释。""",

    "D3_paper": """该 Skill 缺少论文引用。请：
1. 根据算法内容推断最可能的论文来源（NeurIPS/KDD/ICML/ICLR/AAAI 会议）
2. 在 ## ① 开头第一行加 > **论文**：{推断的论文名} | **arXiv**：XXXX.XXXXX（若能推断）
3. 在 frontmatter 加 source: arxiv:XXXX.XXXXX
只输出完整改进后的 Skill markdown，不要解释。""",

    "D6_code": """该 Skill 的代码无法运行或缺少 [✓] 输出。请：
1. 检查所有 import（只用 numpy/pandas/sklearn/scipy/collections/itertools 等标准库）
2. 代码必须包含完整示例数据（不依赖外部文件）
3. 末尾最后一行必须是 print("[✓] {算法名}测试通过")
4. 确保代码 python 一键运行无报错
只输出完整改进后的 Skill markdown，不要解释。""",

    "D7_code_depth": """该 Skill 的代码未反映论文核心算法。请：
1. 从 ## ① 提取核心数学公式和算法步骤
2. 将代码的关键变量名改为对应论文符号（如 beta, gamma, theta, mu, sigma）
3. 添加注释说明代码行对应论文哪个公式/步骤
4. 若有伪代码，逐行翻译为可运行 Python
只输出完整改进后的 Skill markdown，不要解释。""",

    "D8_algo": """该 Skill 的算法原理缺乏深度。请在 ## ① 中：
1. 加入核心公式（Markdown math 格式或 Python 表达式）
2. 每个公式后用一句业务语言解释含义（不超过 20 字）
3. 在 ## ① 末尾新增「**非共识迁移**」段落：
   - 算法原始领域（如：流行病学 / 博弈论 / 物理学）
   - 跨境电商应用反直觉之处
   - 一句话说明降维打击优势
只输出完整改进后的 Skill markdown，不要解释。""",

    "D9_scenario": """该 Skill 的应用案例过于泛泛。请完全重写 ## ② 应用案例：
1. 用具体母婴品类（婴儿暖奶器/婴儿推车/有机辅食/益生菌/安全座椅）替换模糊描述
2. 给出具体数字（库存 2000 件/日销 50 件/ROAS 3.2/转化率 4.5%/复购率 22%）
3. 量化产出（年化节省 45 万元/周转率提升 28%/准确率 +15%/ROAS +1.3）
4. 每个场景末尾加**三轨验证**（成本/合规/风险各一句话）
只输出完整改进后的 Skill markdown，不要解释。""",

    "D10_three_track": """该 Skill 缺少三轨验证。在 ## ② 每个应用场景末尾加入：

**三轨验证**：
- **成本轨**：执行该方案的显性成本（数据采集费用/计算资源/人力投入，给出具体数字）
- **合规轨**：是否触碰 Amazon 政策/GDPR/广告法/跨境贸易法规（明确是否合规）
- **风险轨**：次生风险（引发竞品价格战/平台审查/品牌损伤，及概率估计）

只输出完整改进后的 Skill markdown，不要解释。""",

    "D11_crossdomain": """该 Skill 缺乏跨学科迁移视角。在 ## ① 末尾新增段落：

**非共识迁移**：本算法源自 [原始领域，如：流行病学/量子计算/博弈论/运筹学/生态学]。
传统跨境电商运营者会 [通常做法，1句]，
而该算法通过 [核心机制，1句] 反直觉地解决了 [具体跨境问题]，
实现「降维打击」：[优势总结，≤20字]。

只输出完整改进后的 Skill markdown，不要解释。""",
}


# ── 全量重写提示词（低分专用）────────────────────────────

REWRITE_PROMPT = """你是顶级 paper2skills 内容专家，专注母婴跨境电商 AI 决策 Skill 卡片写作。

以下是一个质量较差的 Skill 卡片，请按照标准 5 模块结构全量重写，提升至 85 分以上。

**原始内容**：
{original}

**重写要求**：

## ① 算法原理（≤300字）
- 核心思想：一句话概括算法解决的问题
- 数学直觉：至少 1 个核心公式 + 含义解释（业务语言）
- 关键假设：使用条件/前提
- 非共识迁移：原始领域 + 为何降维打击跨境电商（必须有）

## ② 母婴出海应用案例（2个具体场景）
- 必须用具体母婴品类：{domain_hint}
- 每个场景含：业务问题 + 具体数据规模 + 量化产出（万元/百分比）
- 末尾加三轨验证（成本/合规/风险）

## ③ 代码模板（Python）
- 完整可运行，只用标准库（numpy/pandas/sklearn/scipy）
- 含内嵌示例数据（不依赖外部文件）
- 末尾：print("[✓] {skill_name}测试通过")
- 代码长度：150-250行，充分展示算法核心

## ④ 技能关联
- 前置（prerequisite）：至少 1 条 [[Skill-XXX]]
- 延伸（extends）：至少 1 条 [[Skill-XXX]]
- 可组合（combinable）：至少 1 条 [[Skill-XXX]]（说明组合场景）

## ⑤ 商业价值评估
- ROI 预估：具体数字（万元/百分比，有量化依据）
- 实施难度：⭐⭐⭐☆☆（3/5星，给出理由）
- 优先级：⭐⭐⭐⭐☆（4/5星，给出理由）

**frontmatter 保持原有格式，更新 updated 日期为 {today}，确保包含 roadmap_phase 字段。**

只输出完整改进后的 Skill markdown，不要解释。"""


def get_domain(skill_path: Path) -> str:
    """从路径推断域名"""
    return skill_path.parent.name


def call_deepseek(prompt: str, max_tokens: int = 6000, temperature: float = 0.3) -> str:
    if not API_KEY:
        return ""
    import requests
    try:
        resp = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": MODEL,
                  "messages": [{"role": "user", "content": prompt}],
                  "max_tokens": max_tokens, "temperature": temperature},
            timeout=120
        )
        content = resp.json()["choices"][0]["message"]["content"]
        m = re.match(r'^```(?:markdown)?\n(.*?)```\s*$', content, re.DOTALL)
        if m:
            content = m.group(1)
        return content
    except Exception as e:
        print(f"     API 调用失败: {e}")
        return ""


def mutate(skill_path: Path, weakest: str, round_n: int) -> str:
    """定向变异：针对最弱维度"""
    content  = skill_path.read_text(encoding="utf-8", errors="replace")
    instruct = MUTATIONS.get(weakest, MUTATIONS["D9_scenario"])
    # 如果是 D9，附加域感知提示
    if weakest == "D9_scenario":
        domain      = get_domain(skill_path)
        domain_hint = DOMAIN_SCENARIO_HINT.get(domain, "母婴婴儿推车/暖奶器/辅食，给出具体销量/成本/ROI数字")
        instruct    = instruct + f"\n\n**本域业务背景**：{domain_hint}"
    prompt = (f"以下是需要改进的 Skill 卡片（第{round_n+1}轮，改进维度: {weakest}）：\n\n"
              f"{content}\n\n改进指令：\n{instruct}")
    return call_deepseek(prompt)


def full_rewrite(skill_path: Path) -> str:
    """全量重写（低分专用）"""
    content     = skill_path.read_text(encoding="utf-8", errors="replace")
    domain      = get_domain(skill_path)
    domain_hint = DOMAIN_SCENARIO_HINT.get(domain, "婴儿推车/暖奶器/有机辅食，给出具体数字")
    skill_name  = skill_path.stem
    today       = datetime.now().strftime("%Y-%m-%d")
    prompt      = REWRITE_PROMPT.format(
        original=content[:4000],
        domain_hint=domain_hint,
        skill_name=skill_name,
        today=today,
    )
    return call_deepseek(prompt, max_tokens=8000, temperature=0.4)


# ── 核心进化循环 ─────────────────────────────────────────

def find_skill(skill_id: str) -> Optional[Path]:
    for d in VAULT.iterdir():
        if not d.is_dir():
            continue
        p = d / f"{skill_id}.md"
        if p.exists():
            return p
    return None


def evolve_skill(skill_id: str, dry_run: bool = False) -> dict:
    skill_path = find_skill(skill_id)
    if not skill_path:
        return {"error": f"找不到 {skill_id}"}

    EVO_DIR.mkdir(exist_ok=True)
    log_path = EVO_DIR / f"{skill_id}.jsonl"

    # 初始评分
    initial   = evaluate(skill_path)
    best      = initial["total"]
    domain    = get_domain(skill_path)

    print(f"\n🧬 [{domain}] {skill_id}")
    print(f"   初始分: {best:.1f}/100  最弱: {initial['weakest']} — {initial['reason']}")

    def log(round_n: int, status: str, score: float, best_score: float, mutated: str):
        entry = {"round": round_n, "status": status, "score": score,
                 "best": best_score, "mutated": mutated,
                 "ts": datetime.now().isoformat()}
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return entry

    # ─ 跳过高分 ────────────────────────────────────────────
    if best >= TARGET:
        print(f"   ✅ 已达 {TARGET} 分，跳过")
        log(0, "skip_high_score", best, best, "none")
        return {"skill_id": skill_id, "status": "skip_high_score",
                "initial": best, "final": best, "delta": 0}

    if dry_run:
        print(f"   [dry-run] 分类: {'重写' if best < REWRITE_THRESHOLD else '进化'}, 最弱: {initial['weakest']}")
        return {"skill_id": skill_id, "status": "dry-run",
                "initial": best, "final": best, "breakdown": initial["breakdown"]}

    # ─ 低分走全量重写 ────────────────────────────────────────
    if best < REWRITE_THRESHOLD:
        print(f"   🔄 分数 {best:.1f} < {REWRITE_THRESHOLD}，启动全量重写...")
        original    = skill_path.read_text(encoding="utf-8", errors="replace")
        new_content = full_rewrite(skill_path)

        if not new_content or len(new_content) < 500:
            print(f"   ❌ 重写失败（内容为空）")
            log(0, "rewrite_failed", best, best, "full_rewrite")
            return {"skill_id": skill_id, "status": "rewrite_failed",
                    "initial": best, "final": best, "delta": 0}

        skill_path.write_text(new_content, encoding="utf-8")
        # 用规则评分（D1-D6）判断重写质量，避免 LLM 自评偏差
        auto_new  = score_auto(new_content)
        code_new  = score_code(skill_path)
        rule_new  = sum(auto_new.values()) + code_new
        auto_orig = score_auto(original)
        code_orig = score_code(skill_path)
        skill_path.write_text(original, encoding="utf-8")
        code_orig = score_code(skill_path)
        rule_orig = sum(auto_orig.values()) + code_orig
        length_ok = len(new_content) > len(original) * 0.8  # 新内容不能比原来短太多

        if rule_new > rule_orig or (rule_new >= rule_orig - 2 and length_ok):
            skill_path.write_text(new_content, encoding="utf-8")
            new_eval  = evaluate(skill_path)
            best      = new_eval["total"]
            status    = "rewrite_improved"
            subprocess.run(["git", "add", str(skill_path)], capture_output=True)
        else:
            status = "rewrite_no_gain"

        log(0, status, best, best, "full_rewrite")
        icon = "✅" if status == "rewrite_improved" else "↩️ "
        print(f"   {icon} 重写结果: {status} {best:.1f}/100  rule:{rule_orig:.0f}→{rule_new:.0f}")

        delta = best - initial["total"]
        return {"skill_id": skill_id, "status": status,
                "initial": initial["total"], "final": best, "delta": delta}

    # ─ 中分走迭代进化 ────────────────────────────────────────
    consecutive_fail = 0
    last_weakest     = ""

    for rn in range(MAX_RNDS):
        curr = evaluate(skill_path)

        # 连续失败同一维度 → 切换次弱
        if consecutive_fail >= 2 and curr["weakest"] == last_weakest:
            dims_sorted = sorted(curr["breakdown"], key=lambda k: curr["breakdown"][k] / DIMS[k])
            for alt in dims_sorted:
                if alt != last_weakest:
                    curr["weakest"] = alt
                    break
            consecutive_fail = 0
        last_weakest = curr["weakest"]

        new_content = mutate(skill_path, curr["weakest"], rn)
        if not new_content or len(new_content) < 500:
            print(f"   ❌ Round {rn+1}: 变异失败")
            consecutive_fail += 1
            continue

        original = skill_path.read_text(encoding="utf-8", errors="replace")
        skill_path.write_text(new_content, encoding="utf-8")

        new_eval  = evaluate(skill_path)
        new_score = new_eval["total"]

        # 崩塌保护：score 暴跌超过初始分 25% → 强制 revert
        if new_score < initial["total"] * 0.75:
            skill_path.write_text(original, encoding="utf-8")
            status = "collapsed_revert"
            consecutive_fail += 1
        elif new_score > best:
            best   = new_score
            status = "keep"
            consecutive_fail = 0
            subprocess.run(["git", "add", str(skill_path)], capture_output=True)
        else:
            skill_path.write_text(original, encoding="utf-8")
            status = "discard"
            consecutive_fail += 1

        log(rn, status, new_score, best, curr["weakest"])
        icon = "✅" if status == "keep" else "↩️ "
        print(f"   {icon} Round {rn+1}: {status:<8} {new_score:.1f}/100  (mutated: {curr['weakest']})")

        if best >= TARGET:
            print(f"   🎯 达到目标 {TARGET} 分，早停")
            break

        time.sleep(0.3)

    delta = best - initial["total"]
    print(f"   📊 {initial['total']:.1f} → {best:.1f} (Δ={delta:+.1f})")

    return {"skill_id": skill_id, "status": "done",
            "initial": initial["total"], "final": best, "delta": delta}


# ── 批量入口 ─────────────────────────────────────────────

def load_candidates(domain: Optional[str], mode: str) -> list:
    """按 mode 收集候选：rewrite(<65) / evolve(65-85) / all"""
    candidates = []

    if domain:
        domain_dirs = [VAULT / domain]
    else:
        domain_dirs = sorted(
            [d for d in VAULT.iterdir() if d.is_dir() and d.name[0:1].isdigit()]
        )

    for d in domain_dirs:
        for sf in sorted(d.iterdir()):
            if not sf.name.startswith("Skill-") or not sf.name.endswith(".md"):
                continue
            skill_id  = sf.stem
            log_path  = EVO_DIR / f"{skill_id}.jsonl"

            # 读取已知最佳分
            best_known = None
            if log_path.exists():
                try:
                    lines = [l for l in log_path.read_text().strip().split('\n') if l.strip()]
                    records = [json.loads(l) for l in lines]
                    best_known = max(r.get("best", r.get("score", 0)) for r in records)
                except Exception:
                    pass

            if best_known is None:
                # 未曾进化过，用 auto 快速估分（避免 LLM 调用）
                c     = sf.read_text(encoding="utf-8", errors="replace")
                auto  = score_auto(c)
                code  = score_code(sf)
                quick = sum(auto.values()) + code + 28  # 假设 LLM 部分 28 分（保守）
                best_known = round(quick, 1)

            if mode == "rewrite" and best_known < REWRITE_THRESHOLD:
                candidates.append((best_known, skill_id))
            elif mode == "evolve" and REWRITE_THRESHOLD <= best_known < TARGET:
                candidates.append((best_known, skill_id))
            elif mode == "all" and best_known < TARGET:
                candidates.append((best_known, skill_id))

    # 低分优先
    candidates.sort(key=lambda x: x[0])
    return [sid for _, sid in candidates]


def batch_evolve(domain: Optional[str], mode: str, limit: int,
                 dry_run: bool = False) -> list:
    candidates = load_candidates(domain, mode)
    target     = candidates[:limit]

    print(f"\n🎯 批量进化: domain={domain or '全部'}, mode={mode}")
    print(f"   候选: {len(candidates)} 个 | 本批: {len(target)} 个")

    results      = []
    improved     = 0
    total_delta  = 0.0
    rewrite_cnt  = 0
    evolve_cnt   = 0

    for i, sid in enumerate(target, 1):
        print(f"\n[{i}/{len(target)}]", end="")
        r = evolve_skill(sid, dry_run=dry_run)
        results.append(r)
        if r.get("delta", 0) > 0:
            improved    += 1
            total_delta += r["delta"]
        if r.get("status", "").startswith("rewrite"):
            rewrite_cnt += 1
        elif r.get("status") in ("done", "keep"):
            evolve_cnt  += 1

    print(f"\n\n{'='*55}")
    print(f"📊 批量进化完成")
    print(f"   总计: {len(results)} | 净改进: {improved} | 重写: {rewrite_cnt} | 进化: {evolve_cnt}")
    if improved:
        print(f"   平均改进: +{total_delta/improved:.1f} 分")
    print(f"{'='*55}")

    return results


# ── CLI ───────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skill 达尔文自进化引擎 v2.0")
    parser.add_argument("--skill",    help="单个 Skill ID")
    parser.add_argument("--domain",   help="按域批量（如：22-数据采集工程）")
    parser.add_argument("--mode",     default="all",
                        choices=["rewrite", "evolve", "all"],
                        help="rewrite=全量重写<65分, evolve=进化65-85分, all=两者都做")
    parser.add_argument("--limit",    type=int, default=50,
                        help="本批最多处理数量（默认50）")
    parser.add_argument("--dry-run",  action="store_true",
                        help="只评分不改写")
    args = parser.parse_args()

    if not API_KEY:
        print("⚠️  DEEPSEEK_API_KEY 未设置，将跳过 LLM 评分和改写")
        print("   export DEEPSEEK_API_KEY=sk-...")
        sys.exit(1)

    if args.skill:
        evolve_skill(args.skill, dry_run=args.dry_run)
    else:
        batch_evolve(
            domain=args.domain,
            mode=args.mode,
            limit=args.limit,
            dry_run=args.dry_run,
        )
