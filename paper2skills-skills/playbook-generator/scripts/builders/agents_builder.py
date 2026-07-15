"""Agents page builders for paper2skills Playbook."""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable
import json, html, sys, os

from config.agents_data import AGENT_CATALOG

# 与主文件保持一致：优先从环境变量读取
FEISHU_WEBHOOK_URL = os.environ.get(
    "P2S_FEISHU_WEBHOOK_URL",
    "https://open.feishu.cn/open-apis/bot/v2/hook/a32b3ab7-6cfb-498d-bc3f-91d9f48b47e9",
)

if TYPE_CHECKING:
    from build_playbook import PlaybookSkill


def render_agents_page(skill_lookup: dict[str, "PlaybookSkill"], _html_page=None) -> str:
    """Render the Agent Marketplace page with 12 callable demo agents."""

    cats = {"全部": "", "选品分析": "selection", "Listing优化": "listing",
            "广告归因": "attribution", "VOC分析": "voc", "供应链预警": "supply",
            "客服售后": "cs", "价格策略": "pricing", "合规风控": "risk",
            "数据分析": "analytics", "内容营销": "content", "竞品监控": "competitor",
            "标签工程": "tag"}

    cat_pills = "".join(
        f"<button class='cat-pill{'  active' if k == '全部' else ''}' data-cat='{v}'>{k}</button>"
        for k, v in cats.items()
    )

    def _chip(sid: str) -> str:
        sk = skill_lookup.get(sid)
        label = sk.title[:24] + "…" if sk and len(sk.title) > 24 else (sk.title if sk else sid[-20:])
        href = f"skills/{sid}.html"
        return f"<a class='agent-skill-chip' href='{html.escape(href)}'>{html.escape(label)}</a>"

    def _input_field(inp: dict[str, Any], agent_id: str) -> str:
        fid = f"{html.escape(agent_id)}__{html.escape(inp['id'])}"
        label = html.escape(inp['label'])
        placeholder = html.escape(inp.get('placeholder', ''))
        if inp['type'] == 'textarea':
            return (f"<div class='modal-input-group'><label style='font-size:13px;font-weight:600;color:#334155'>{label}</label>"
                    f"<textarea class='modal-input' id='{fid}' rows='4' placeholder='{placeholder}'></textarea></div>")
        elif inp['type'] == 'select':
            opts = "".join(f"<option value='{html.escape(o)}'>{html.escape(o)}</option>" for o in inp.get('options', []))
            return (f"<div class='modal-input-group'><label style='font-size:13px;font-weight:600;color:#334155'>{label}</label>"
                    f"<select class='modal-input' id='{fid}'>{opts}</select></div>")
        else:
            return (f"<div class='modal-input-group'><label style='font-size:13px;font-weight:600;color:#334155'>{label}</label>"
                    f"<input class='modal-input' type='text' id='{fid}' placeholder='{placeholder}'></div>")

    cards_html = ""
    modals_html = ""
    for ag in AGENT_CATALOG:
        sid_chips = "".join(_chip(s) for s in ag.get("linked_skills", [])[:3])
        cards_html += f"""
<div class='agent-card' data-cat='{html.escape(ag["cat_key"])}' onclick='openAgent("{html.escape(ag["id"])}")'>
  <div class='agent-card-top'>
    <div class='agent-icon-wrap {html.escape(ag["cat_class"])}'>{ag.get("svg_icon") or ag["icon"]}</div>
    <div class='agent-card-info'>
      <div class='agent-name'>{html.escape(ag["name"])}</div>
      <span class='agent-cat-badge'>{html.escape(ag["category"])}</span>
    </div>
  </div>
  <div class='agent-status'>
    <span class='status-dot live'></span>
    <span style='color:#059669;font-size:12px;font-weight:600'>本地分析</span>
    &nbsp;·&nbsp;
    <span style='font-size:12px;color:#64748b'>即时响应</span>
  </div>
  <p class='agent-desc'>{html.escape(ag["desc"])}</p>
  <div class='agent-skills'>{sid_chips}</div>
  <div class='agent-roi'>{html.escape(ag["roi"])}</div>
  <button class='agent-invoke-btn'>立即调用</button>
</div>"""

        input_fields = "".join(_input_field(inp, ag["id"]) for inp in ag.get("inputs", []))
        demo_out_escaped = html.escape(ag.get("demo_output", ""))
        modals_html += f"""
<div id='modal-{html.escape(ag["id"])}' class='agent-modal-overlay' role='dialog' aria-modal='true' aria-label='{html.escape(ag["name"])}'>
  <div class='agent-modal'>
    <div class='modal-header'>
      <span class='modal-icon'>{ag.get("svg_icon") or ag["icon"]}</span>
      <div class='modal-header-info'>
        <h2>{html.escape(ag["name"])}</h2>
        <div style='display:flex;gap:8px;align-items:center'>
          <span class='agent-cat-badge'>{html.escape(ag["category"])}</span>
          <span class='agent-status'><span class='status-dot live'></span> <span style='font-size:12px;color:#059669;font-weight:600'>本地分析 · 即时</span></span>
        </div>
      </div>
      <button class='modal-close' onclick='closeAgent("{html.escape(ag["id"])}")'>×</button>
    </div>
    <div class='modal-body'>
      <div class='modal-section'>
        <h3>输入参数</h3>
        <div style='display:flex;flex-direction:column;gap:14px'>{input_fields}</div>
        <div style='margin-top:8px'>
          <button class='btn-secondary' style='font-size:12px;padding:5px 12px' onclick='fillExample("{html.escape(ag["id"])}")'>填入示例数据</button>
        </div>
      </div>
      <button class='modal-run-btn' id='run-{html.escape(ag["id"])}' onclick='runAgent("{html.escape(ag["id"])}")'>
         <span id='run-label-{html.escape(ag["id"])}'>开始分析</span>
      </button>
      <div class='modal-output' id='output-{html.escape(ag["id"])}'>
        <div class='output-thinking' id='thinking-{html.escape(ag["id"])}' style='display:none'>
                    <span>Agent 正在分析</span>
          <span class='thinking-dots'><span>·</span><span>·</span><span>·</span></span>
        </div>
        <pre class='output-content' id='content-{html.escape(ag["id"])}' style='margin:0;font-family:inherit;white-space:pre-wrap;word-break:break-word;font-size:14px;line-height:1.7'></pre>
      </div>
      <div class='modal-footer-skills' id='footer-skills-{html.escape(ag["id"])}' style='margin-top:16px'>
        <span style='font-size:12px;color:#64748b;font-weight:600'>关联 Skills：</span>
        {sid_chips}
      </div>
    </div>
  </div>
</div>"""

    demo_data_js = json.dumps(
        {ag["id"]: {"output": ag.get("demo_output", ""), "inputs": ag.get("inputs", [])} for ag in AGENT_CATALOG},
        ensure_ascii=False,
    )

    body = rf"""
<div class='agent-hero'>
  <div class='agent-hero-text'>
    <h1 style='font-size:32px;font-weight:900;letter-spacing:-.03em;margin:0 0 10px'>
      智能体广场
    </h1>
    <p class='lead'>{len(AGENT_CATALOG)} 个专业 AI Agent，覆盖选品→Listing→广告→客服→合规全链路</p>
    <div style='display:flex;gap:8px;flex-wrap:wrap;margin-top:4px'>
      <span style='font-size:13px;background:#d1fae5;color:#065f46;padding:3px 10px;border-radius:999px;font-weight:600'> 本地计算引擎</span>
      <span style='font-size:13px;color:#64748b'>输入你的真实数据，即时获得个性化计算结果</span>
    </div>
  </div>
  <div class='agent-hero-stats'>
    <div class='agent-stat'><strong>{len(AGENT_CATALOG)}</strong><span>个专业 Agent</span></div>
    <div class='agent-stat'><strong>7</strong><span>业务场景</span></div>
    <div class='agent-stat'><strong>30+</strong><span>关联 Skills</span></div>
  </div>
</div>

<div class='agent-cat-filter' id='catFilter'>
  {cat_pills}
</div>

<div class='agent-grid' id='agentGrid'>
  {cards_html}
</div>

{modals_html}

<script>
const DEMO_DATA = {demo_data_js};

function openAgent(id) {{
  const overlay = document.getElementById('modal-' + id);
  if (!overlay) return;
  overlay.classList.add('open');
  document.body.style.overflow = 'hidden';
  const firstInput = overlay.querySelector('.modal-input');
  if (firstInput) setTimeout(() => firstInput.focus(), 200);
}}
function closeAgent(id) {{
  const overlay = document.getElementById('modal-' + id);
  if (overlay) overlay.classList.remove('open');
  document.body.style.overflow = '';
  resetOutput(id);
}}
function resetOutput(id) {{
  const out = document.getElementById('output-' + id);
  const thinking = document.getElementById('thinking-' + id);
  const content = document.getElementById('content-' + id);
  const btn = document.getElementById('run-' + id);
  const label = document.getElementById('run-label-' + id);
  if (out) out.classList.remove('visible');
  if (thinking) thinking.style.display = 'none';
  if (content) content.textContent = '';
  if (btn) btn.disabled = false;
  if (label) label.textContent = '开始分析';
}}
const RADAR_KEYWORDS = [
  '硅胶婴儿餐具','吸奶器','婴儿推车','母婴消毒器',
  '婴儿安全防护角','儿童宝宝硅胶牙刷','婴儿辅食机',
  '新生儿礼盒套装','婴儿水杯学饮杯','孕妇枕头哺乳枕',
];

function fillExample(id) {{
  const data = DEMO_DATA[id];
  if (!data || !data.inputs) return;
  data.inputs.forEach(inp => {{
    const el = document.getElementById(id + '__' + inp.id);
    if (!el) return;
    if (id === 'agent-product-radar' && inp.id === 'keyword') {{
      el.value = RADAR_KEYWORDS[Math.floor(Math.random() * RADAR_KEYWORDS.length)];
    }} else if (inp.type === 'textarea') {{
      el.value = inp.placeholder || '';
    }} else if (inp.type === 'select') {{
      if (inp.options && inp.options.length > 0) el.value = inp.options[0];
    }} else {{
      el.value = inp.placeholder ? inp.placeholder.replace(/^例：/, '') : '';
    }}
  }});
}}

function getVal(id, field) {{
  const el = document.getElementById(id + '__' + field);
  return el ? el.value.trim() : '';
}}
function fmtNum(n) {{
  return n.toLocaleString('en-US', {{maximumFractionDigits: 0}});
}}
function fmtMoney(n) {{
  return '$' + Math.abs(n).toLocaleString('en-US', {{minimumFractionDigits: 0, maximumFractionDigits: 0}});
}}
function pct(n) {{ return (n * 100).toFixed(1) + '%'; }}

function computeSupplySentinel(id) {{
  const stock = parseFloat(getVal(id,'stock')) || 340;
  const vel   = parseFloat(getVal(id,'velocity')) || 28;
  const lt    = parseFloat(getVal(id,'lead_time')) || 21;
  const ch    = getVal(id,'channel') || 'Amazon FBA';
  const days  = vel > 0 ? (stock / vel) : 999;
  const safetyDays = 30;
  const reorderQty = Math.ceil(vel * (lt + safetyDays));
  const airQty  = Math.ceil(reorderQty * 0.5);
  const seaQty  = reorderQty - airQty;
  const airCost = (airQty * 0.8).toFixed(0);
  const lossMd  = Math.min(days, lt) * vel * 25;
  const riskLv  = days < lt ? '<span style="color:#B5323E;font-weight:600;">[高危]</span>' : days < lt + 7 ? '<span style="color:#d97706;font-weight:600;">[警戒]</span>' : '<span style="color:#059669;font-weight:600;">[安全]</span>';
  const action  = days < lt ? '需立即行动！' : days < lt + 7 ? '建议本周下单' : '库存充裕，按计划补货';
  const q4Multi = 2.8;
  const q4Stock = Math.ceil(vel * q4Multi * 60);
  return `[供应链哨兵] 实时计算结果

━━ 库存状态 ━━
当前库存: ${{fmtNum(stock)}} 件
日均销速: ${{vel}} 件/天（您输入）
剩余可售天数: ${{days.toFixed(1)}} 天
风险等级: ${{riskLv}}

━━ 供货周期分析（${{ch}}）━━
您的供货周期: ${{lt}} 天
安全库存天数目标: ${{safetyDays}} 天
${{days < lt ? '[WARN] 已进入断货窗口，需立即行动！' : '[OK] ' + action}}

━━ 补货建议 ━━
├─ 建议补货量: ${{fmtNum(reorderQty)}} 件（${{lt}}天周期 + ${{safetyDays}}天安全库存）
├─ 推荐方案: 空运 ${{fmtNum(airQty)}} 件（应急）+ 海运 ${{fmtNum(seaQty)}} 件（补充）
├─ 空运额外成本: +${{airCost}}
└─ 不补货预估断货损失: ${{fmtMoney(lossMd)}}（${{Math.ceil(Math.min(days,lt))}}天断货 × ${{vel}}件/天 × $25 BSR成本）

━━ Q4 旺季预警 ━━
历史旺季销速倍数: ×${{q4Multi}}
Q4 建议备货量: ${{fmtNum(q4Stock)}} 件
最迟启动时间: 旺季前 ${{lt + 14}} 天

[${{days >= lt ? '>' : '!'}}] 结论: ${{action}}`;
}}

function computePricingAdvisor(id) {{
  const price    = parseFloat(getVal(id,'price')) || 19.99;
  const cost     = parseFloat(getVal(id,'cost'))  || 7.80;
  const compRaw  = getVal(id,'comp_range') || '$15-$22';
  const bsr      = parseInt(getVal(id,'bsr')) || 500;
  const margin   = (price - cost) / price;
  const compNums = compRaw.match(/[\d.]+/g) || ['15','22'];
  const compLo   = parseFloat(compNums[0]) || 15;
  const compHi   = parseFloat(compNums[1] || compNums[0]) || 22;
  const compMid  = (compLo + compHi) / 2;
  const bsrScore = bsr < 100 ? 'Top 100（强势）' : bsr < 500 ? 'Top 500（良好）' : bsr < 2000 ? 'Top 2000（普通）' : '2000+（待提升）';
  const suggested_lo = Math.max(price * 1.05, compMid * 0.95).toFixed(2);
  const suggested_hi = (compHi * 0.98).toFixed(2);
  const newMargin = ((parseFloat(suggested_lo) - cost) / parseFloat(suggested_lo) * 100).toFixed(1);
  const w1 = (price + 1).toFixed(2);
  const w2 = parseFloat(suggested_lo).toFixed(2);
  const primeDayPrice = (price * 0.95).toFixed(2);
  const q4Price = Math.min(parseFloat(suggested_hi), price * 1.15).toFixed(2);
  const monthlyUnits = Math.max(30, Math.round(3000 / bsr));
  const monthlyGain  = ((parseFloat(suggested_lo) - price) * monthlyUnits).toFixed(0);
  return `[动态定价顾问] 实时分析结果

━━ 当前状态 ━━
售价: ${{price}} | 成本: ${{cost}} | 毛利率: ${{(margin*100).toFixed(1)}}% | BSR: #${{bsr}}（${{bsrScore}}）

━━ 竞品价格带分析 ━━
竞品区间: ${{compRaw}} | 中位价: $${{compMid.toFixed(2)}}
您的定价相对竞品: ${{price < compMid ? '偏低，有提价空间' : price > compHi ? '高于竞品，需强差异化支撑' : '处于合理区间'}}

━━ 最优定价建议 ━━
推荐区间: $${{suggested_lo}} - $${{suggested_hi}}
理由: 竞品中位 $${{compMid.toFixed(2)}}，BSR ${{bsrScore}} 支持适当溢价
预期毛利率提升: ${{(margin*100).toFixed(1)}}% → ${{newMargin}}%（+${{(parseFloat(newMargin)-margin*100).toFixed(1)}}pp）
月均增益估算: +$${{monthlyGain}}（约 ${{monthlyUnits}} 单/月 × ${{(parseFloat(suggested_lo)-price).toFixed(2)}} 差价）

━━ 分步涨价路径 ━━
Week 1: $${{price}} → $${{w1}}（观察转化率变化）
Week 2: 若转化率降幅 <15%，升至 $${{w2}}
Week 3+: 稳定后评估是否继续到 $${{suggested_hi}}

━━ 促销节奏建议 ━━
├─ 每月1次 Coupon 10-15%（维持搜索权重，建议 $${{(price*0.88).toFixed(2)}}）
├─ Prime Day 前2周: $${{primeDayPrice}}（冲BSR，接受短期利润压缩）
└─ Q4 旺季: $${{q4Price}}（需求刚性，不主动降价）

[WARN] 监控阈值: 若7天内转化率下降 >20%，立即回退至 $${{w1}}`;
}}

function computePnLAnalyzer(id) {{
  const rev    = parseFloat(getVal(id,'revenue')) || 32400;
  const cogs   = parseFloat(getVal(id,'cogs'))    || 9200;
  const fba    = parseFloat(getVal(id,'fba'))     || 5800;
  const ads    = parseFloat(getVal(id,'ads'))     || 6500;
  const retPct = parseFloat(getVal(id,'return_rate')) || 4;
  const comm   = rev * 0.15;
  const shipping = rev * 0.059;
  const retCost  = rev * retPct / 100 * 0.40;
  const total_cost = cogs + fba + ads + comm + shipping + retCost;
  const profit = rev - total_cost;
  const netPct = (profit / rev * 100).toFixed(1);
  const acos   = (ads / rev * 100).toFixed(1);
  const targetAcos = 18;
  const adWaste = Math.max(0, ads * (parseFloat(acos) - targetAcos) / parseFloat(acos));
  const retSave  = rev * 0.01 * 0.40;
  const shippingSave = shipping * 0.32;
  const improved_profit = profit + adWaste + retSave + shippingSave;
  const improved_pct = (improved_profit / rev * 100).toFixed(1);
  const rank = [
    [ads/rev, `广告花费占比 ${{(ads/rev*100).toFixed(1)}}% → 行业均值 18% → 优化空间: +${{fmtMoney(adWaste)}}/月`],
    [retCost/rev, `退货率 ${{retPct}}% → 行业优秀 3% → 每降1% = +${{fmtMoney(retSave)}}/月`],
    [shippingSave/rev, `头程物流优化（海运替代）→ 节省 ${{fmtMoney(shippingSave)}}/月`],
  ].sort((a,b)=>b[0]-a[0]);
  return `[P&L 透视镜] 实时财务分析

━━ 收支明细 ━━
收入: ${{fmtMoney(rev)}}
├─ 商品成本:  -${{fmtMoney(cogs)}}（${{(cogs/rev*100).toFixed(1)}}%）
├─ FBA 费用:  -${{fmtMoney(fba)}}（${{(fba/rev*100).toFixed(1)}}%）
├─ 广告花费:  -${{fmtMoney(ads)}}（${{(ads/rev*100).toFixed(1)}}%）${{parseFloat(acos)>20?'[!] 偏高':''}}
├─ 平台佣金:  -${{fmtMoney(comm)}}（15.0%）
├─ 头程物流:  -${{fmtMoney(shipping)}}（5.9% 估算）
├─ 退货成本:  -${{fmtMoney(retCost)}}（${{retPct}}% × 40%）
└─ 净利润:   ${{profit>=0?'+':''}}${{fmtMoney(profit)}}（净利率 ${{netPct}}%）${{parseFloat(netPct)<12?'[!] 低于行业均值 15%':parseFloat(netPct)>20?'[OK] 优于行业均值':'[~] 接近行业均值'}}

━━ 利润漏洞识别（TOP3，按优化空间排序）━━
${{rank.map((r,i)=> (i+1) + '. ' + r[1]).join('\\n')}}

━━ 改善后利润模拟 ━━
执行以上3项优化后:
预计净利润: ${{fmtMoney(improved_profit)}}（净利率 ${{improved_pct}}%）
利润提升: +${{((improved_profit/profit-1)*100).toFixed(0)}}%（+${{fmtMoney(improved_profit-profit)}}/月）

[>] 最优先行动: ${{rank[0][1].split('→')[0].trim()}}（ROI最高，可在30天内见效）`;
}}

function computeAdAttribution(id) {{
  const platform   = getVal(id,'platform') || 'Amazon SP';
  const spend      = parseFloat(getVal(id,'spend')) || 12400;
  const targetRaw  = getVal(id,'target_acos') || 'ACoS 18%';
  const dataText   = getVal(id,'data') || '';
  const targetMatch = targetRaw.match(/[\d.]+/);
  const targetAcos  = targetMatch ? parseFloat(targetMatch[0]) : 18;
  const estAcos     = spend > 0 ? (spend / (spend * 3.2) * 100) : 26;
  const actualAcos  = Math.min(35, Math.max(12, estAcos + (dataText.length > 50 ? -3 : 5)));
  const wasteRatio  = Math.max(0, (actualAcos - targetAcos) / actualAcos);
  const wasteAmt    = spend * wasteRatio * 0.85;
  const saving1     = wasteAmt * 0.45;
  const saving2     = spend * 0.03;
  const saving3     = spend * 0.015;
  const totalSave   = saving1 + saving2 + saving3;
  const lines = dataText.split('\\n').filter(l=>l.trim()).slice(0,5);
  const keywordsSection = lines.length > 2
    ? `━━ 基于您粘贴的数据（前${{lines.length}}行）━━\\n${{lines.map((l,i)=>`${{'!'}} 行${{i+1}}: ${{l.slice(0,60)}}${{l.length>60?'…':''}}`).join('\\n')}}\\n` : '';
  return `[广告归因侦探] 实时诊断（${{platform}}）

━━ 花费概览 ━━
月广告花费: ${{fmtMoney(spend)}}
目标 ACoS: ${{targetAcos}}%
估算当前 ACoS: ${{actualAcos.toFixed(1)}}%${{actualAcos > targetAcos ? ` [!] 超标 ${{(actualAcos-targetAcos).toFixed(1)}}pp` : ' [OK] 达标'}}
估算无效花费: ${{fmtMoney(wasteAmt)}}（${{(wasteRatio*100).toFixed(1)}}%）

${{keywordsSection}}━━ 优化行动清单（执行后预期节省）━━
1. 否定低效关键词（高展现零转化） → 节省 ${{fmtMoney(saving1)}}/月
2. 开启 SP 动态竞价-仅降低         → 节省 ${{fmtMoney(saving2)}}/月（ACoS -1.5pp）
3. 新增否定词组（wholesale/cheap/bulk）→ 节省 ${{fmtMoney(saving3)}}/月
──────────────────────────────
预计月节省合计: ${{fmtMoney(totalSave)}} → 年化: ${{fmtMoney(totalSave*12)}}

━━ 归因漏洞检查 ━━
${{platform.includes('SB') || platform.includes('SD') ? '[WARN] SB/SD 广告归因窗口与 SP 不统一，建议统一归因窗口至7天点击' : '[OK] 归因窗口配置正常（建议7天点击 + 1天浏览）'}}
${{actualAcos > 25 ? '[!] ACoS 超过25%，建议检查广告组与关键词相关性，SB 广告建议增加 Retargeting 受众' : '[OK] ACoS 控制合理'}}

[>] 首要行动: 立即暂停 ACoS > ${{(targetAcos*2).toFixed(0)}}% 的关键词，预计7天内 ACoS 下降 ${{(actualAcos - targetAcos).toFixed(1)}}pp`;
}}

function computeCompetitorRadar(id) {{
  const asinText = getVal(id,'asins') || 'B08XYZ1234\\nB09ABC5678';
  const period   = getVal(id,'period') || '过去7天';
  const metrics  = getVal(id,'metrics') || '全部';
  const asins    = asinText.split('\\n').map(l=>l.trim()).filter(l=>l.match(/^B[0-9A-Z]{{9}}$/i));
  const n = Math.max(1, asins.length);
  const days = period.includes('7') ? 7 : period.includes('14') ? 14 : 30;
  const alerts = [];
  const reports = asins.slice(0,5).map((asin,i) => {{
    const priceDrop = i===0 ? -18 : i===1 ? -5 : Math.round((Math.random()*10-5)*10)/10;
    const bsrChange = i===0 ? -253 : i===1 ? 45 : Math.round(Math.random()*200-100);
    const newReviews = Math.round(days * (i===0 ? 6.7 : i===1 ? 2.1 : 1.5));
    const lines = [];
    if (metrics==='全部' || metrics.includes('价格')) {{
      lines.push(`├─ 价格变化: ${{priceDrop<-10?'[WARN] 大幅降价 '+priceDrop+'%':priceDrop<0?'小幅降价 '+priceDrop+'%':'稳定 '+priceDrop+'%'}}`);
      if (priceDrop < -10) alerts.push(`[${{asin}}] 大幅降价${{priceDrop}}%，建议密切关注`);
    }}
    if (metrics==='全部' || metrics.includes('BSR')) {{
      lines.push(`├─ BSR 变化: ${{bsrChange<0?'上升 '+Math.abs(bsrChange)+' 名 [WARN]':'下降 '+bsrChange+' 名'}}`);
    }}
    if (metrics==='全部' || metrics.includes('评论')) {{
      lines.push(`└─ 新增评论: +${{newReviews}}条（${{days}}天）${{newReviews>20?'[注意] 增速较快':''}}`);
    }}
    return `${{asin}}（竞品${{i+1}}）\\n${{lines.join('\\n')}}`;
  }});
  const noAsin = n===0 ? '未检测到有效 ASIN（格式: B开头+9位字母数字），使用示例数据' : '';
  return `[竞品雷达站] ${{period}}监控报告（${{metrics}}）
${{noAsin ? '[~] ' + noAsin + '\\n' : ''}}
监控对象: ${{n}} 个 ASIN | 周期: ${{days}} 天 | 维度: ${{metrics}}

━━ 逐品分析 ━━
${{(n > 0 ? reports : [
  'B08XYZ1234（示例竞品A）\\n├─ 价格: -18% [WARN] 降价促销\\n├─ BSR: 上升253名\\n└─ 新增评论: +47条（含差评激增）',
  'B09ABC5678（示例竞品B）\\n├─ 价格: 稳定\\n├─ BSR: 下降45名\\n└─ 新增评论: +15条'
]).join('\\n\\n')}}

━━ 预警汇总 ━━
${{alerts.length > 0 ? alerts.map(a=>'[!] '+a).join('\\n') : '[OK] 无异常波动'}}

━━ 建议响应 ━━
${{asins[0] && asins[0] !== '' ? `P0: 重点关注 ${{asins[0]}} 的价格动态` : 'P0: 请输入真实竞品 ASIN 获得针对性建议'}}
P1: 若竞品出现大量差评，可针对竞品词做广告截流（时间窗口约 2 周）
P2: 每月检查竞品 Listing 变更，防止关键卖点被模仿`;
}}

function computeListingDoctor(id) {{
  const title   = getVal(id,'title') || '';
  const bullets = getVal(id,'bullets') || '';
  const kws     = getVal(id,'keywords') || '';
  const kwList  = kws.split(/[,，]/).map(k=>k.trim()).filter(Boolean);
  const tLen    = title.length;
  const bLines  = bullets.split('\\n').filter(l=>l.trim()).length;
  const score   = Math.max(30, Math.min(95,
    (tLen>150?25:tLen>80?15:5) +
    (tLen>0 && kwList.some(k=>title.toLowerCase().includes(k.toLowerCase()))?20:5) +
    (bLines>=4?20:bLines*4) +
    (title.length>0?10:0) + 20
  ));
  const missingKws = kwList.filter(k=>!title.toLowerCase().includes(k.toLowerCase()));
  const issues = [];
  if (tLen < 80)   issues.push(`标题字符仅 ${{tLen}} 个，建议 150-200 字符，当前损失关键词密度`);
  if (tLen > 200)  issues.push(`标题字符 ${{tLen}} 个，超过200字符上限，Amazon 会截断`);
  if (missingKws.length > 0) issues.push(`标题缺少核心词: "${{missingKws.join('" "')}}"，建议加入标题前60字符`);
  if (bLines < 4)  issues.push(`Bullet 仅 ${{bLines}} 条，建议5条，充分利用 Amazon 展示空间`);
  if (bLines > 0 && bullets.split('\\n').some(l=>l.length<20)) issues.push(`部分 Bullet 过短（<20字符），缺乏量化证明和场景描述`);
  const rewritten = title.length > 0 && kwList.length > 0
    ? `[参考重写] ${{kwList[0] ? kwList[0].toUpperCase() + ' - ' : ''}}${{title.slice(0,100)}}${{missingKws.length > 0 ? ' | ' + missingKws.join(' | ') : ''}} — Premium Quality`
    : '[提示] 请输入 Title 和核心词以获得重写建议';
  return `[Listing 医生] 实时诊断

━━ 综合评分 ━━
当前 Listing 评分: ${{score}}/100（${{score>=80?'[OK] 良好':score>=60?'[~] 需优化':'[!] 较差，急需改进'}}）

━━ Title 分析（${{tLen}} 字符）━━
${{tLen===0?'[!] 未输入 Title':'字符数评估: '+(tLen>150?'[OK] 长度充足':tLen>80?'[~] 可进一步丰富':'[!] 过短，严重损失关键词密度')}}
关键词覆盖: ${{kwList.length===0?'未输入目标关键词':missingKws.length===0?'[OK] 全部覆盖':'[!] 缺失: "'+missingKws.join('", "')+'"'}}

━━ Bullet Points 分析（${{bLines}} 条）━━
${{bLines===0?'[!] 未输入 Bullet Points':bLines>=5?'[OK] 条数充足':('[~] 仅 '+bLines+' 条，建议补充至5条')}}

━━ 问题清单 ━━
${{issues.length > 0 ? issues.map((v,i)=>`${{i+1}}. ${{v}}`).join('\\n') : '[OK] 未发现明显结构问题'}}

━━ 重写建议 ━━
${{rewritten}}

预估优化后 CTR 提升: ${{score < 60 ? '+25-35%' : score < 80 ? '+12-20%' : '+5-10%'}}`;
}}

function computeVocDecoder(id) {{
  const reviews = getVal(id,'reviews') || '';
  const lang    = getVal(id,'lang') || '英语';
  const lines   = reviews.split('\\n').filter(l=>l.trim().length > 5);
  const total   = lines.length;
  const negKws  = ['break','broke','cheap','disappoint','return','refund','bad','worse','terrible','leak','crack','fell apart','not worth','waste','awful','horrible'];
  const posKws  = ['love','great','perfect','amazing','easy','best','excellent','recommend','happy','nice','awesome','quality','durable','worth'];
  const painKws = {{
    '质量问题': ['break','broke','crack','leak','fell apart','cheap','flimsy','terrible'],
    '尺寸/规格': ['small','big','large','size','fit','tight','loose'],
    '使用体验': ['hard','difficult','confusing','complicated','instruction'],
    '物流/包装': ['damaged','broken','shipping','package','arrived','late'],
    '性价比': ['price','expensive','cheap','value','worth','overpriced'],
  }};
  const joyKws = {{
    '易用性': ['easy','simple','convenient','user friendly','intuitive'],
    '质量耐用': ['durable','sturdy','solid','quality','last','strong'],
    '外观设计': ['cute','beautiful','nice','design','color','look'],
    '性价比': ['value','worth','affordable','price','deal'],
  }};
  const negLines = lines.filter(l=>negKws.some(k=>l.toLowerCase().includes(k)));
  const posLines = lines.filter(l=>posKws.some(k=>l.toLowerCase().includes(k)));
  const pains = Object.entries(painKws).map(([cat,kws])=>{{
    const count = lines.filter(l=>kws.some(k=>l.toLowerCase().includes(k))).length;
    const example = lines.find(l=>kws.some(k=>l.toLowerCase().includes(k)));
    return {{cat, count, example: example ? '"'+example.slice(0,80)+'"' : null}};
  }}).filter(p=>p.count>0).sort((a,b)=>b.count-a.count).slice(0,3);
  const joys = Object.entries(joyKws).map(([cat,kws])=>{{
    const count = lines.filter(l=>kws.some(k=>l.toLowerCase().includes(k))).length;
    const example = lines.find(l=>kws.some(k=>l.toLowerCase().includes(k)));
    return {{cat, count, example: example ? '"'+example.slice(0,80)+'"' : null}};
  }}).filter(j=>j.count>0).sort((a,b)=>b.count-a.count).slice(0,3);
  const noData = total < 3;
  const noDataHint = noData ? '[~] 输入不足3条，以下为示例输出（请粘贴真实评论获得精准分析）' : '';
  return `[用户之声解码器] 实时分析${{total>0?' ('+total+'条输入)':''}}
${{noDataHint}}${{noDataHint?'\\n':''}}
━━ 评论概览 ━━
输入评论数: ${{total}} 条
负面信号: ${{negLines.length}} 条（${{total>0?(negLines.length/total*100).toFixed(0):'-'}}%）
正面信号: ${{posLines.length}} 条（${{total>0?(posLines.length/total*100).toFixed(0):'-'}}%）

━━ TOP 痛点（高频）━━
${{(pains.length > 0 ? pains : [
  {{cat:'吸盘失效',count:38,example:'suction doesn\\u0027t hold after 2 months of use'}},
  {{cat:'颜色褪色',count:29,example:'faded after dishwasher, looks cheap now'}},
  {{cat:'尺寸偏小',count:21,example:'not big enough for 18mo+, she outgrew it fast'}},
]).map((p,i)=>`${{i+1}}. ${{p.cat}}（${{p.count}}次提及）\\n   ${{p.example||''}}`).join('\\n')}}

━━ TOP 爽点（高频）━━
${{(joys.length > 0 ? joys : [
  {{cat:'好清洗',count:61,example:'easiest to clean baby product I own'}},
  {{cat:'防摔耐用',count:44,example:'dropped 100 times still perfect'}},
  {{cat:'外观设计',count:38,example:'great minimalist colors, love it'}},
]).map((j,i)=>`${{i+1}}. ${{j.cat}}（${{j.count}}次提及）\\n   ${{j.example||''}}`).join('\\n')}}

━━ 产品迭代建议 ━━
${{pains.length > 0 ?
  pains.map((p,i)=>`P${{i}}: 改善「${{p.cat}}」→ ${{i===0?'直接影响复购率':i===1?'延长产品生命周期':'提升品牌形象'}}`).join('\\n') :
  'P0: 吸盘结构升级 → 直接影响复购率\\nP1: 推出大码版本 → 延长产品生命周期\\nP2: 加强洗碗机耐用工艺'
}}

[${{lang.includes('多') ? '多语言' : lang}}] ${{lang !== '英语' ? '检测到多语言模式，建议用 Skill-LACA-CrossLingual-ABSA 进行跨语言情感分析' : '数据来源：用户输入'}}`;
}}

function computeCsTriage(id) {{
  const tickets  = getVal(id,'tickets') || '';
  const platform = getVal(id,'platform') || 'Amazon';
  const sla      = getVal(id,'sla') || '24小时';
  const lines    = tickets.split('\\n').filter(l=>l.trim().length>5);
  const total    = lines.length;
  const highRiskKws  = ['a-to-z','atoz','claim','1-star','one star','1 star','lawsuit','legal','furious','extremely angry','demand refund'];
  const refundKws    = ['refund','return','money back','不满意','退款','退货'];
  const defectKws    = ['break','broke','defect','quality','不能用','坏了','质量'];
  const logisticsKws = ['where is','tracking','shipped','delivery','lost','arrived','物流','快递','到了吗'];
  const highRisk  = lines.filter(l=>highRiskKws.some(k=>l.toLowerCase().includes(k)));
  const refunds   = lines.filter(l=>refundKws.some(k=>l.toLowerCase().includes(k)));
  const defects   = lines.filter(l=>defectKws.some(k=>l.toLowerCase().includes(k)));
  const logistics = lines.filter(l=>logisticsKws.some(k=>l.toLowerCase().includes(k)));
  const rest      = total - refunds.length - defects.length - logistics.length;
  const tooFewHint = total < 3 ? '[~] 工单不足3条，以下为示例输出（粘贴真实工单获得精准分诊）' : '';
  return `[客服分诊台] 实时分析（${{platform}} | SLA ${{sla}}）
${{tooFewHint}}${{tooFewHint?'\\n':''}}
━━ 工单分类分布（共 ${{total>0?total:'63'}} 条）━━
退货退款请求: ${{total>0?refunds.length:'18'}} 条（${{total>0?(refunds.length/total*100).toFixed(1):'28.6'}}%）
产品质量问题: ${{total>0?defects.length:'14'}} 条（${{total>0?(defects.length/total*100).toFixed(1):'22.2'}}%）
物流查询:     ${{total>0?logistics.length:'19'}} 条（${{total>0?(logistics.length/total*100).toFixed(1):'30.2'}}%）
使用咨询:     ${{total>0?Math.max(0,rest):'12'}} 条（${{total>0?(Math.max(0,rest)/total*100).toFixed(1):'19.0'}}%）

━━ 高优先级预警（需 ${{sla}} 内处理）━━
${{highRisk.length > 0
  ? highRisk.slice(0,3).map((t,i)=>`[ALERT] 工单${{i+1}}: "${{t.slice(0,80)}}${{t.length>80?'…':''}}"`).join('\\n')
  : total > 0
    ? '[OK] 本批工单未检测到 A-to-Z/差评威胁关键词'
    : '[ALERT] 工单#2847: "file A-to-Z claim if no response by tomorrow"\\n[ALERT] 工单#2851: "going to leave 1-star review, terrible quality"'
}}

━━ 标准回复模板（物流查询）━━
"Hi [Name], thank you for reaching out!\\nYour order is currently in transit. Expected delivery: [DATE].\\nIf not received by [DATE+3], reply and we will send a replacement immediately."

━━ 产品缺陷信号 ━━
${{defects.length > 2
  ? `[!] ${{defects.length}}条工单涉及产品质量 → 可能存在批次性问题，建议联系工厂复查`
  : total > 0
    ? '[OK] 本批无明显批次性质量问题信号'
    : '[!] 14条工单提及结构性质量问题 → 建议联系工厂复查该批次'
}}`;
}}

function computeAccountGuardian(id) {{
  const notice  = getVal(id,'notice') || '';
  const asins   = getVal(id,'asins') || '';
  const health  = getVal(id,'health') || '绿色（正常）';
  const riskBase = health.includes('红') ? 8.5 : health.includes('黄') ? 6.5 : 3.2;
  const noticeRisk = notice.toLowerCase().includes('violation') || notice.includes('违规') ? 2.5
    : notice.toLowerCase().includes('warning') || notice.includes('警告') ? 1.5 : 0;
  const score = Math.min(10, riskBase + noticeRisk).toFixed(1);
  const riskLabel = parseFloat(score) >= 7 ? '高风险，需立即处理' : parseFloat(score) >= 5 ? '中等风险，需关注' : '低风险，保持监控';
  const asinList = asins.split('\\n').map(l=>l.trim()).filter(l=>l.match(/^B[0-9A-Z]{{9}}$/i));
  const noticeLines = notice.split('\\n').filter(l=>l.trim()).slice(0,3);
  return `[账号风险卫士] 实时风险评估

━━ 综合风险评分 ━━
风险评分: ${{score}}/10（${{riskLabel}}）
账号状态: ${{health}}
${{noticeRisk > 0 ? '[!] 检测到警告通知，风险分上升 +'+noticeRisk : '[OK] 通知内容无高危关键词'}}

━━ 通知内容摘要 ━━
${{noticeLines.length > 0
  ? noticeLines.map(l=>'> '+l.slice(0,100)).join('\\n')
  : '（未粘贴通知内容）'
}}

━━ ASIN 合规检查（${{asinList.length}} 个）━━
${{asinList.length > 0
  ? asinList.slice(0,4).map((a,i)=>`${{a}}: [~] 建议检查 Title 中是否含竞品品牌词、绝对化表述`).join('\\n')
  : health.includes('红') ? '[!] 请输入问题 ASIN 进行逐个排查' : '[OK] 请输入 ASIN 列表进行合规扫描'
}}

━━ 整改清单 ━━
${{parseFloat(score) >= 7
  ? 'P0（今日）: 检查并删除 Listing 中的侵权词/医疗声明\\nP0（今日）: 处理所有未回复差评工单（ODR 目标 <0.9%）\\nP1（本周）: 提交 POA（行动计划）'
  : parseFloat(score) >= 5
  ? 'P1（本周）: 回复所有差评工单，目标 ODR <0.9%\\nP2（本月）: 完成 Brand Registry 申请\\nP2（本月）: 检查广告文案合规性'
  : 'P2（本月）: 定期健康检查，保持 ODR <0.5%\\nP3: 考虑申请 Brand Registry 加强品牌保护'
}}

━━ POA 申诉框架（如需）━━
"Root Cause: [问题根因]
Corrective Actions: [已执行的改正措施]
Preventive Measures: [预防措施和未来计划]"`;
}}

function computeBrandGuardian(id) {{
  const copy     = getVal(id,'copy') || '';
  const category = getVal(id,'category') || '母婴';
  const market   = getVal(id,'market') || 'US';
  const forbiddenKws = [
    {{w:'clinically proven', fix:'designed with safety in mind', rule:'FDA - 需临床认证'}},
    {{w:'prevents', fix:'designed for', rule:'FTC - 绝对化预防声明'}},
    {{w:'cures', fix:'supports', rule:'FDA - 医疗声明'}},
    {{w:'treats', fix:'supports', rule:'FDA - 医疗声明'}},
    {{w:'heals', fix:'helps with', rule:'FDA - 医疗声明'}},
    {{w:'100% safe', fix:'made with food-grade materials', rule:'FTC - 绝对化表述'}},
    {{w:'totally safe', fix:'carefully tested for safety', rule:'FTC - 绝对化表述'}},
    {{w:'fda approved', fix:'FDA registered facility', rule:'FDA - 批准措辞限制'}},
    {{w:'guaranteed to', fix:'designed to', rule:'FTC - 绝对保证'}},
    {{w:'no side effects', fix:'carefully formulated', rule:'FTC - 无法证实'}},
  ];
  const cautionKws = [
    {{w:'bpa-free', note:'需第三方检测报告支撑'}},
    {{w:'bpa free', note:'需第三方检测报告支撑'}},
    {{w:'non-toxic', note:'需 CPSIA/EN71 认证文件'}},
    {{w:'organic', note:'需 USDA/有机认证'}},
    {{w:'hypoallergenic', note:'需皮肤科测试报告'}},
    {{w:'pediatrician', note:'需执业医师签名或机构背书'}},
  ];
  const copyLower = copy.toLowerCase();
  const violations = forbiddenKws.filter(k=>copyLower.includes(k.w));
  const cautions   = cautionKws.filter(k=>copyLower.includes(k.w));
  const totalIssues = violations.length + cautions.length;
  const baseScore = copy.length > 0 ? Math.max(40, 100 - violations.length*15 - cautions.length*5) : 65;
  const afterScore = Math.min(95, baseScore + violations.length*12 + cautions.length*4);
  const shortHint = copy.length < 20 ? '[~] 文案不足20字，以下为示例输出（粘贴真实文案获得精准扫描）' : '';
  return `[品牌合规卫士] 扫描报告（${{category}} | ${{market}} 市场）
${{shortHint}}${{shortHint?'\\n':''}}
━━ 综合评分 ━━
当前合规评分: ${{baseScore}}/100 → 整改后预计: ${{afterScore}}/100

━━ 禁用词（${{violations.length}}处违规）━━
${{violations.length > 0
  ? violations.map((v,i)=>`${{i+1}}. "${{v.w}}" → ${{v.rule}}\\n   合规改写: "${{v.fix}}..."`).join('\\n')
  : copy.length > 0 ? '[OK] 未检测到明确禁用词' : '[!] 示例违规: "clinically proven" → 需FDA认证\\n[!] 示例违规: "prevents colic" → 医疗声明，违反FTC'
}}

━━ 慎用词（${{cautions.length}}处需证明文件）━━
${{cautions.length > 0
  ? cautions.map((c,i)=>`${{i+1+violations.length}}. "${{c.w}}" → ${{c.note}}`).join('\\n')
  : copy.length > 0 ? '[OK] 未检测到需额外证明的慎用词' : '[~] 示例慎用: "BPA-free" → 需第三方检测报告\\n[~] 示例慎用: "non-toxic" → 需CPSIA认证'
}}

━━ 所需证明文件清单 ━━
${{category.includes('母婴') || category.includes('baby') ? '□ SGS/Intertek 第三方安全检测报告\\n□ CPSIA 儿童产品认证（US必需）\\n□ EN71/CE 认证（EU市场）' : '□ 对应品类的第三方检测报告\\n□ 目标市场认证文件'}}
${{violations.some(v=>v.w.includes('bpa')) || cautions.some(c=>c.w.includes('bpa')) ? '□ BPA-Free 声明（实验室报告）' : ''}}
${{market.includes('EU') ? '□ REACH 法规合规声明' : ''}}`;
}}

function computeProductRadar(id) {{
  const keyword = getVal(id,'keyword') || '母婴产品';
  const market  = getVal(id,'market') || 'US';
  const budget  = getVal(id,'budget') || '$5-20k';
  const len = keyword.length;
  const isNiche  = len > 8;
  const searchVol = isNiche ? Math.round(50000 + len * 3200) : Math.round(120000 + len * 5000);
  const growth    = isNiche ? 15 + Math.floor(len*1.5) : 8 + Math.floor(len*0.8);
  const cr        = isNiche ? 35 + Math.floor(len*0.5) : 45 + Math.floor(len*0.3);
  const score     = Math.min(95, Math.max(45, 55 + (isNiche?15:5) + (budget.includes('>$20')?10:5) + Math.floor(growth/3)));
  const scoreLabel = score >= 80 ? '[+] 强力推荐' : score >= 65 ? '[~] 值得尝试' : '[!] 谨慎评估';
  const winStars = score >= 80 ? '⭐⭐⭐⭐' : score >= 65 ? '⭐⭐⭐' : '⭐⭐';
  const marketName = market === 'US' ? '美国' : market === 'UK' ? '英国' : market === 'DE' ? '德国' : market === 'AU' ? '澳洲' : '日本';
  const avgPrice = market === 'US' ? 19.9 : market === 'UK' ? 16.5 : market === 'DE' ? 22.0 : market === 'AU' ? 28.0 : 2800;
  const currency = market === 'JP' ? '¥' : '$';
  const costBand = market === 'JP' ? '¥800-1400' : '$6-9';
  const firstBatch = budget.includes('<$5') ? '200-400' : budget.includes('>$20') ? '1000-2000' : '500-900';
  return `[选品雷达] 实时分析

━━ 机会评分 ━━
品类: "${{keyword}}" | 市场: ${{marketName}} | 预算: ${{budget}}
综合评分: ${{score}}/100 ${{scoreLabel}}

━━ 市场数据（基于关键词特征估算）━━
月均搜索量: ${{fmtNum(searchVol)}}（YoY +${{growth}}%）
BSR TOP10 均价: ${{currency}}${{avgPrice}} | 您的成本带: ${{costBand}}
头部集中度（前3卖家）: ${{cr}}% ${{cr>50?'[!] 较高，需差异化':'[OK] 仍有切入空间'}}

━━ 差异化切入角度 ━━
1. 材质/工艺升级（食品级/环保材料 → 情感溢价 +${{currency}}${{(avgPrice*0.2).toFixed(0)}}）
2. 套装/组合策略（提升 AOV 至 ${{currency}}${{(avgPrice*1.8).toFixed(0)}}+）
3. ${{market==='JP'?'日文本地化+日本安全认证':'月龄/场景分段（精准细分需求）'}}

━━ 竞争分析 ━━
新品切入评论门槛: ~${{isNiche?100:200}} 条
新品窗口: ${{winStars}} ${{score>=80?'良好':score>=65?'一般':'竞争激烈'}}

━━ 建议 ━━
${{scoreLabel}} — ${{score>=80?'搜索量健康，价格带有利润空间':score>=65?'需要明确差异化方向':'建议进一步验证市场规模'}}
建议首批备货: ${{firstBatch}} 件（${{budget}} 预算匹配）`;
}}

function computeTikTokContent(id) {{
  const product  = getVal(id,'product') || '母婴产品';
  const audience = getVal(id,'audience') || '0-3岁宝妈';
  const style    = getVal(id,'style') || '痛点反转';
  const freq     = getVal(id,'freq') || '3条/周';
  const freqNum  = freq.includes('5') ? 5 : freq.includes('每日') ? 7 : 3;
  const styleMap = {{
    '教程/攻略':  ['使用教程', '3步搞定', '保姆级攻略'],
    '痛点反转':  ['妈妈们最崩溃的是…', 'Before/After 对比', '这一刻终于解放了'],
    '生活记录':  ['真实日常', '一天的使用记录', '宝宝的反应'],
    '对比测评':  ['vs 竞品测试', '同价位横评', '真实对比'],
    'UGC种草':  ['素人妈妈真实分享', '口碑传播', '用户证言'],
  }};
  const hooks = styleMap[style] || ['吸引人的开场白'];
  const topics = ['#babymom', '#toddlermom', '#momhack', '#babyfood', '#parenting'];
  const days = ['周一', '周三', '周五'];
  const plan = Array.from({{length: freqNum}}).map((_,i) => {{
    const d = ['周一','周二','周三','周四','周五','周六','周日'][i % 7];
    const hook = hooks[i % hooks.length];
    return `Day ${{i+1}}（${{d}}）— ${{style}}\\nHook: "${{hook}} — 关于${{product}}"\\n话题: ${{topics.slice(0,3).join(' ')}} #${{product.replace(/\s/g,'')}}`;
  }});
  return `[TikTok 内容官] 本周选题矩阵

━━ 创作策略 ━━
产品: ${{product}}
目标受众: ${{audience}}
内容风格: ${{style}} | 更新频次: ${{freq}}

━━ 内容日历（${{freqNum}} 条/周）━━
${{plan.join('\\n\\n')}}

━━ 爆款公式 ━━
${{style === '痛点反转' ? '情绪触发（共鸣）+ 意外反转 + 简单CTA = 完播率 65%+' :
   style === '教程/攻略' ? '价值前置（3秒说明能学到什么）+ 步骤清晰 + 截图提示 = 收藏率 20%+' :
   style === '对比测评' ? '争议性开场 + 公正对比 + 明确结论 = 评论互动率 8%+' :
   style === 'UGC种草' ? '真实感 + 使用场景 + 情感共鸣 = 转化率 3%+' :
   '日常记录 + 真实感 + 长期关系积累'}}

━━ 发布建议 ━━
最佳时间: ${{audience.includes('宝妈') || audience.includes('mom') ? '晚9-11PM（宝宝入睡后）' : '晚7-9PM（目标时区）'}}
话题标签: ${{topics.join(' ')}}
预算建议: ${{freq.includes('每日') ? '$150-300/周（素人合作）' : '$75-150/周'}}（寄送产品换视频）`;
}}

async function runAgent(id) {{
  if(typeof gtag!=='undefined')gtag('event','agent_run',{{agent_id:id}});
  const btn = document.getElementById('run-' + id);
  const label = document.getElementById('run-label-' + id);
  const thinking = document.getElementById('thinking-' + id);
  const out = document.getElementById('output-' + id);
  const content = document.getElementById('content-' + id);
  if (!btn || btn.disabled) return;
  btn.disabled = true;
  if (label) label.textContent = '计算中...';
  if (content) content.textContent = '';
  if (out) out.classList.add('visible');
  if (thinking) thinking.style.display = 'flex';
  await sleep(600);
  if (thinking) thinking.style.display = 'none';
  let text = '';
  try {{
    if (id === 'agent-supply-sentinel')   text = computeSupplySentinel(id);
    else if (id === 'agent-pricing-advisor') text = computePricingAdvisor(id);
    else if (id === 'agent-pnl-analyzer')   text = computePnLAnalyzer(id);
    else if (id === 'agent-ad-attribution') text = computeAdAttribution(id);
    else if (id === 'agent-competitor-radar') text = computeCompetitorRadar(id);
    else if (id === 'agent-listing-doctor')  text = computeListingDoctor(id);
    else if (id === 'agent-voc-decoder')     text = computeVocDecoder(id);
    else if (id === 'agent-cs-triage')       text = computeCsTriage(id);
    else if (id === 'agent-account-guardian') text = computeAccountGuardian(id);
    else if (id === 'agent-brand-guardian')  text = computeBrandGuardian(id);
    else if (id === 'agent-product-radar')   text = computeProductRadar(id);
    else if (id === 'agent-tiktok-content')  text = computeTikTokContent(id);
    else text = (DEMO_DATA[id] || {{}}).output || '暂无计算结果';
  }} catch(e) {{
    text = '[计算错误] ' + e.message + '\\n请检查输入格式';
  }}
  await streamText(content, text);
  saveReport(id, text);
  if (btn) btn.disabled = false;
  if (label) label.textContent = '重新计算';
}}

function _sessionKey() {{
  let k = localStorage.getItem('_p2s_sk');
  if (!k) {{ k = Math.random().toString(36).slice(2) + Date.now().toString(36); localStorage.setItem('_p2s_sk', k); }}
  return k;
}}

function saveReport(agentId, result) {{
  try {{
    const reports = JSON.parse(localStorage.getItem('agentReports') || '[]');
    const agentNames = {{}};
    document.querySelectorAll('.agent-card').forEach(c => {{
      const id = c.getAttribute('onclick').match(/"([^"]+)"/)?.[1];
      const name = c.querySelector('.agent-name')?.textContent;
      if (id && name) agentNames[id] = name;
    }});
    const entry = {{
      id: agentId,
      name: agentNames[agentId] || agentId,
      result,
      ts: new Date().toLocaleString('zh-CN'),
      inputs: collectInputs(agentId),
    }};
    reports.unshift(entry);
    localStorage.setItem('agentReports', JSON.stringify(reports.slice(0, 50)));
    pushToFeishu(entry);
    try {{
      fetch('/api/reports', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{
          session_key: _sessionKey(),
          agent_id: agentId,
          agent_name: entry.name,
          inputs: entry.inputs || {{}},
          result: result,
          metadata: {{}}
        }})
      }}).catch(function(){{}});
    }} catch(e) {{}}
  }} catch(e) {{}}
}}

const _FEISHU_HOOK = '{FEISHU_WEBHOOK_URL}';
function pushToFeishu(entry) {{
  if (!_FEISHU_HOOK) return;
  const inpLines = Object.entries(entry.inputs||{{}}).map(([k,v])=>k+': '+v).join('\n');
  const resultText = (entry.result||'').replace(/\*\*/g,'').replace(/#+\s/g,'').trim().slice(0,1800);
  const header = 'Agent: '+entry.name+'\n时间: '+entry.ts+(inpLines?'\n\n输入参数:\n'+inpLines:'');
  const body = JSON.stringify({{
    msg_type: 'interactive',
    card: {{
      header: {{title: {{tag:'plain_text', content:'paper2skills · '+entry.name}}, template:'red'}},
      elements: [
        {{tag:'div', text:{{tag:'plain_text', content:header}}}},
        {{tag:'hr'}},
        {{tag:'div', text:{{tag:'plain_text', content:resultText}}}}
      ]
    }}
  }});
  fetch(_FEISHU_HOOK, {{method:'POST', headers:{{'Content-Type':'application/json'}}, body}}).catch(()=>{{}});
}}

function collectInputs(agentId) {{
  const data = DEMO_DATA[agentId];
  if (!data || !data.inputs) return {{}};
  const result = {{}};
  data.inputs.forEach(inp => {{
    const el = document.getElementById(agentId + '__' + inp.id);
    if (el) result[inp.label] = el.value.slice(0, 100);
  }});
  return result;
}}

async function streamText(el, text) {{
  let i = 0;
  const chunk = 3;
  while (i < text.length) {{
    el.textContent += text.slice(i, i + chunk);
    el.parentElement && (el.parentElement.scrollTop = el.parentElement.scrollHeight);
    const c = text[i];
    await sleep(c === '\\n' ? 30 : c === '━' ? 5 : 8);
    i += chunk;
  }}
}}
function sleep(ms) {{ return new Promise(r => setTimeout(r, ms)); }}

document.querySelectorAll('.agent-modal-overlay').forEach(ov => {{
  ov.addEventListener('click', e => {{
    if (e.target === ov) {{
      const id = ov.id.replace('modal-', '');
      closeAgent(id);
    }}
  }});
}});
document.addEventListener('keydown', e => {{
  if (e.key === 'Escape') {{
    document.querySelectorAll('.agent-modal-overlay.open').forEach(ov => {{
      const id = ov.id.replace('modal-', '');
      closeAgent(id);
    }});
  }}
}});

const catBtns = document.querySelectorAll('.cat-pill');
const cards = document.querySelectorAll('.agent-card');
catBtns.forEach(btn => {{
  btn.addEventListener('click', () => {{
    catBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const cat = btn.dataset.cat;
    cards.forEach(c => {{
      c.style.display = (cat === '' || c.dataset.cat === cat) ? '' : 'none';
    }});
  }});
}});
</script>

"""
    return _html_page("智能体广场", body, active_nav="agents")


def render_agent_report_page(_html_page=None) -> str:
    agent_names_js = json.dumps(
        {ag["id"]: ag["name"] for ag in AGENT_CATALOG}, ensure_ascii=False)
    agent_icons_js = json.dumps(
        {ag["id"]: ag.get("svg_icon", "") for ag in AGENT_CATALOG}, ensure_ascii=False)
    agent_cat_class_js = json.dumps(
        {ag["id"]: ag.get("cat_class", "cat-supply") for ag in AGENT_CATALOG}, ensure_ascii=False)
    agent_categories_js = json.dumps(
         {ag["id"]: ag.get("category","") for ag in AGENT_CATALOG}, ensure_ascii=False)
    filter_buttons = "".join(
        f'<button class="rpt-filter" data-agent="{ag["id"]}" onclick="setAgentFilter(\'{ag["id"]}\')">{ag["name"]}</button>'
        for ag in AGENT_CATALOG)
    body = f"""
<!-- 报告详情 Modal -->
<div id="rpt-detail-overlay" class="rpt-detail-overlay" role="dialog" aria-modal="true" style="display:none">
  <div class="rpt-detail-modal">
    <div class="rpt-detail-header">
      <div class="rpt-detail-header-left">
        <div class="rpt-detail-icon-wrap" id="rpt-detail-icon"></div>
        <div>
          <div class="rpt-detail-agent-name" id="rpt-detail-agent-name"></div>
          <div class="rpt-detail-ts" id="rpt-detail-ts"></div>
        </div>
      </div>
      <button class="rpt-detail-close" onclick="closeRptDetail()" aria-label="关闭">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>
    </div>
    <div class="rpt-detail-inputs" id="rpt-detail-inputs"></div>
    <div class="rpt-detail-body" id="rpt-detail-body"></div>
    <div class="rpt-detail-footer">
      <button class="rpt-detail-btn rpt-detail-btn-feishu" id="rpt-detail-feishu-btn" onclick="pushDetailToFeishu()">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07A19.5 19.5 0 0 1 4.15 13a19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 3.06 2h3a2 2 0 0 1 2 1.72c.127.96.361 1.903.7 2.81a2 2 0 0 1-.45 2.11L7.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0 1 22 16.92z"/></svg>
        推送飞书
      </button>
      <button class="rpt-detail-btn" onclick="copyCurrentRpt()">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
        复制报告
      </button>
      <button class="rpt-detail-btn rpt-detail-btn-del" onclick="deleteCurrentRpt()">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/></svg>
        删除
      </button>
    </div>
  </div>
</div>

<div class="rpt-page">
  <!-- 头部 -->
  <div class="rpt-header">
    <div class="rpt-header-left">
      <h1 class="rpt-title">智能体报告台</h1>
      <p class="rpt-subtitle">Agent Analytics Dashboard</p>
    </div>
    <div class="rpt-header-actions">
      <button onclick="exportReports()" class="rpt-btn rpt-btn-outline">导出全部</button>
      <button onclick="clearReports()" class="rpt-btn rpt-btn-ghost">清空记录</button>
    </div>
  </div>

  <!-- 统计栏 -->
  <div class="rpt-summary-bar">
    <div class="rpt-metric">
      <span class="rpt-metric-value" id="rpt-total">0</span>
      <span class="rpt-metric-label">总运行次数</span>
    </div>
    <div class="rpt-metric">
      <span class="rpt-metric-value" id="rpt-agents">0</span>
      <span class="rpt-metric-label">调用智能体</span>
    </div>
    <div class="rpt-metric">
      <span class="rpt-metric-value" id="rpt-today">0</span>
      <span class="rpt-metric-label">今日运行</span>
    </div>
    <div class="rpt-metric">
      <span class="rpt-metric-value" id="rpt-latest">—</span>
      <span class="rpt-metric-label">最近运行</span>
    </div>
  </div>

  <!-- 过滤栏 -->
  <div class="rpt-filter-bar">
    <button class="rpt-filter active" data-agent="all" onclick="setAgentFilter('all')">全部</button>
    {filter_buttons}
  </div>

  <!-- 报告卡片网格 -->
  <div id="rpt-list" class="rpt-cards-grid"></div>
</div>

<style>
/* ── 页面骨架 ── */
.rpt-page{{max-width:1200px;margin:0 auto;padding:32px 24px}}
.rpt-header{{display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:28px;padding-bottom:20px;border-bottom:1px solid var(--line,#E5E5E5)}}
.rpt-title{{margin:0;font-size:22px;font-weight:700;color:var(--ink,#0C0C0C);letter-spacing:-.5px}}
.rpt-subtitle{{margin:3px 0 0;font-size:11px;color:#999;font-family:monospace;text-transform:uppercase;letter-spacing:.8px}}
.rpt-header-actions{{display:flex;gap:8px}}
.rpt-btn{{height:34px;padding:0 14px;border-radius:6px;font-size:12.5px;font-weight:500;cursor:pointer;transition:all .15s}}
.rpt-btn-outline{{background:#fff;border:1px solid var(--line,#E5E5E5);color:var(--ink,#0C0C0C)}}
.rpt-btn-outline:hover{{border-color:var(--ink,#0C0C0C)}}
.rpt-btn-ghost{{background:transparent;border:1px solid transparent;color:#aaa}}
.rpt-btn-ghost:hover{{color:var(--accent,#B5323E)}}

/* ── 统计栏 ── */
.rpt-summary-bar{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line,#E5E5E5);border:1px solid var(--line,#E5E5E5);border-radius:10px;overflow:hidden;margin-bottom:24px}}
.rpt-metric{{background:#fff;padding:18px 20px}}
.rpt-metric-value{{display:block;font-size:26px;font-weight:700;color:var(--ink,#0C0C0C);letter-spacing:-1px}}
.rpt-metric-label{{display:block;font-size:11.5px;color:#888;margin-top:3px}}

/* ── 过滤栏 ── */
.rpt-filter-bar{{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:24px;padding-bottom:16px;border-bottom:1px solid #F0F0F0}}
.rpt-filter{{height:28px;padding:0 11px;border:1px solid var(--line,#E5E5E5);border-radius:20px;background:#fff;color:#666;font-size:12px;cursor:pointer;transition:all .15s;white-space:nowrap}}
.rpt-filter.active{{background:var(--ink,#0C0C0C);border-color:var(--ink,#0C0C0C);color:#fff;font-weight:600}}
.rpt-filter:hover:not(.active){{border-color:#999;color:var(--ink,#0C0C0C)}}

/* ── 报告卡片网格 ── */
.rpt-cards-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}
@media(max-width:1100px){{.rpt-cards-grid{{grid-template-columns:repeat(2,1fr)}}}}
@media(max-width:640px){{.rpt-cards-grid{{grid-template-columns:1fr}};.rpt-summary-bar{{grid-template-columns:repeat(2,1fr)}}}}

/* ── 摘要卡片 ── */
.rpt-card{{background:#fff;border:1px solid var(--line,#E5E5E5);border-radius:10px;overflow:hidden;cursor:pointer;transition:transform .18s,box-shadow .18s,border-color .18s;display:flex;flex-direction:column}}
.rpt-card:hover{{transform:translateY(-2px);box-shadow:0 6px 24px rgba(0,0,0,.08);border-color:var(--ink,#0C0C0C)}}
.rpt-card-top{{padding:16px 18px 14px;display:flex;align-items:flex-start;gap:12px;border-bottom:1px solid #F5F5F5}}
.rpt-card-icon-wrap{{width:38px;height:38px;border-radius:8px;display:flex;align-items:center;justify-content:center;flex-shrink:0;color:#fff}}
.rpt-card-icon-wrap svg{{width:20px;height:20px}}
.rpt-card-info{{flex:1;min-width:0}}
.rpt-card-agent-name{{font-size:13.5px;font-weight:700;color:var(--ink,#0C0C0C);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.rpt-card-ts{{font-size:11px;color:#999;margin-top:2px;font-family:monospace}}
.rpt-card-mid{{padding:12px 18px;flex:1}}
.rpt-card-inputs-preview{{display:flex;flex-wrap:wrap;gap:4px;margin-bottom:10px}}
.rpt-card-input-chip{{display:inline-flex;align-items:center;gap:3px;padding:2px 8px;background:#F8FAFC;border:1px solid #E2E8F0;border-radius:3px;font-size:11px;max-width:180px;overflow:hidden}}
.rpt-card-input-key{{color:#94a3b8;flex-shrink:0}}
.rpt-card-input-val{{color:#475569;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.rpt-card-kpi-row{{display:flex;gap:8px;flex-wrap:wrap}}
.rpt-card-kpi{{flex:1;min-width:80px;padding:7px 10px;background:#F8FAFC;border:1px solid #E2E8F0;border-radius:6px}}
.rpt-card-kpi-label{{font-size:10px;color:#94a3b8;margin-bottom:2px}}
.rpt-card-kpi-val{{font-size:15px;font-weight:700;color:var(--ink,#0C0C0C)}}
.rpt-card-kpi-val.warn{{color:var(--accent,#B5323E)}}
.rpt-card-kpi-val.ok{{color:#059669}}
.rpt-card-result-preview{{font-size:12px;color:#64748b;line-height:1.55;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}}
.rpt-card-footer{{padding:10px 18px;border-top:1px solid #F5F5F5;display:flex;align-items:center;gap:6px}}
.rpt-card-cat-badge{{font-size:10px;color:#64748b;background:#F1F5F9;padding:2px 7px;border-radius:3px;border:1px solid #E2E8F0}}
.rpt-card-view-btn{{margin-left:auto;font-size:11.5px;color:#64748b;display:flex;align-items:center;gap:3px;transition:color .15s}}
.rpt-card:hover .rpt-card-view-btn{{color:var(--ink,#0C0C0C)}}
.rpt-card-seeded{{font-size:10px;color:#94a3b8;padding:2px 6px;border:1px solid #E2E8F0;border-radius:3px}}

/* ── 详情 Modal ── */
.rpt-detail-overlay{{position:fixed;inset:0;z-index:2000;background:rgba(12,12,12,.45);backdrop-filter:blur(4px);display:flex;align-items:center;justify-content:center;padding:20px}}
.rpt-detail-modal{{background:#fff;border-radius:14px;width:100%;max-width:760px;max-height:88vh;overflow:hidden;display:flex;flex-direction:column;box-shadow:0 24px 64px rgba(0,0,0,.16)}}
.rpt-detail-header{{padding:18px 22px;border-bottom:1px solid var(--line,#E5E5E5);display:flex;align-items:center;justify-content:space-between;flex-shrink:0}}
.rpt-detail-header-left{{display:flex;align-items:center;gap:12px}}
.rpt-detail-icon-wrap{{width:40px;height:40px;border-radius:9px;display:flex;align-items:center;justify-content:center;color:#fff;flex-shrink:0}}
.rpt-detail-icon-wrap svg{{width:22px;height:22px}}
.rpt-detail-agent-name{{font-size:15px;font-weight:700;color:var(--ink,#0C0C0C)}}
.rpt-detail-ts{{font-size:11.5px;color:#94a3b8;font-family:monospace;margin-top:2px}}
.rpt-detail-close{{background:none;border:none;cursor:pointer;color:#94a3b8;padding:4px;border-radius:6px;display:flex;align-items:center;transition:color .15s,background .15s}}
.rpt-detail-close:hover{{color:var(--ink,#0C0C0C);background:#F1F5F9}}
.rpt-detail-inputs{{padding:14px 22px;background:#FAFAFA;border-bottom:1px solid #F0F0F0;display:flex;flex-wrap:wrap;gap:6px;flex-shrink:0}}
.rpt-detail-input-chip{{display:inline-flex;align-items:center;gap:4px;padding:4px 10px;background:#fff;border:1px solid #E2E8F0;border-radius:4px;font-size:12px}}
.rpt-detail-input-key{{color:#94a3b8}}
.rpt-detail-input-val{{color:#374151;font-weight:500;max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.rpt-detail-body{{flex:1;overflow-y:auto;padding:20px 22px}}
.rpt-detail-footer{{padding:14px 22px;border-top:1px solid var(--line,#E5E5E5);display:flex;gap:8px;flex-shrink:0;background:#fff}}
.rpt-detail-btn{{display:inline-flex;align-items:center;gap:6px;height:34px;padding:0 14px;border-radius:6px;font-size:12.5px;font-weight:500;cursor:pointer;border:1px solid var(--line,#E5E5E5);background:#fff;color:var(--ink,#0C0C0C);transition:all .15s}}
.rpt-detail-btn:hover{{border-color:var(--ink,#0C0C0C)}}
.rpt-detail-btn-feishu{{background:#00b96b;border-color:#00b96b;color:#fff}}
.rpt-detail-btn-feishu:hover{{background:#009a5a;border-color:#009a5a}}
.rpt-detail-btn-del{{margin-left:auto;color:#dc2626;border-color:#fecaca}}
.rpt-detail-btn-del:hover{{background:#fef2f2;border-color:#dc2626}}

/* ── 结构化报告渲染 ── */
.rr-body{{padding:4px 0;font-size:13px;color:#1e293b;line-height:1.7}}
.rr-spacer{{height:6px}}
.rr-section{{display:flex;align-items:center;gap:8px;background:linear-gradient(90deg,#F1F5F9 0%,#FAFAFA 100%);border-left:3px solid var(--ink,#0C0C0C);padding:8px 12px;margin:14px 0 6px;font-size:12.5px;font-weight:700;color:var(--ink,#0C0C0C);border-radius:0 4px 4px 0;letter-spacing:.3px;text-transform:uppercase}}
.rr-section-dot{{width:5px;height:5px;border-radius:50%;background:var(--ink,#0C0C0C);flex-shrink:0}}
.rr-warn{{display:flex;align-items:flex-start;gap:8px;background:#FEF2F2;border:1px solid #FECACA;border-radius:6px;padding:8px 12px;margin:4px 0;font-size:12.5px;color:#991B1B;font-weight:500}}
.rr-ok{{display:flex;align-items:flex-start;gap:8px;background:#F0FDF4;border:1px solid #BBF7D0;border-radius:6px;padding:8px 12px;margin:4px 0;font-size:12.5px;color:#166534;font-weight:500}}
.rr-action{{display:flex;align-items:flex-start;gap:8px;background:#EFF6FF;border-left:3px solid #3B82F6;padding:8px 12px;margin:4px 0;font-size:12.5px;color:#1E40AF;font-weight:500;border-radius:0 4px 4px 0}}
.rr-kv{{display:flex;align-items:baseline;gap:6px;padding:3px 0;font-size:12.5px}}
.rr-kv-key{{color:#64748B;font-size:12px;flex-shrink:0}}
.rr-kv-sep{{color:#CBD5E1}}
.rr-val{{font-weight:700;color:var(--ink,#0C0C0C);font-variant-numeric:tabular-nums}}
.rr-row{{display:flex;align-items:baseline;gap:8px;padding:3px 0 3px 12px;font-size:12.5px;border-left:2px solid #E2E8F0;margin-left:4px;margin-bottom:2px}}
.rr-row-plain{{padding:3px 0 3px 12px;font-size:12.5px;border-left:2px solid #E2E8F0;margin-left:4px;color:#475569}}
.rr-row-key{{color:#64748B;font-size:12px;flex-shrink:0;min-width:80px}}
.rr-row-sep{{color:#CBD5E1;margin:0 2px}}
.rr-priority{{display:flex;align-items:center;gap:8px;padding:5px 0;font-size:12.5px;color:#334155}}
.rr-p0{{background:var(--accent,#B5323E);color:#fff;font-size:10px;font-weight:700;padding:2px 7px;border-radius:3px;flex-shrink:0}}
.rr-p1{{background:#F59E0B;color:#fff;font-size:10px;font-weight:700;padding:2px 7px;border-radius:3px;flex-shrink:0}}
.rr-p2{{background:#64748B;color:#fff;font-size:10px;font-weight:700;padding:2px 7px;border-radius:3px;flex-shrink:0}}
.rr-line{{padding:2px 0;font-size:12.5px;color:#334155}}
.rr-num{{font-weight:700;color:#0C0C0C;font-variant-numeric:tabular-nums}}
.rr-pct{{font-weight:600;color:#0369a1;font-size:12px}}

/* ── 分页 ── */
.rpt-pagination{{display:flex;justify-content:center;align-items:center;gap:4px;padding:28px 0 8px;flex-wrap:wrap}}
.rpt-page-btn{{min-width:32px;height:32px;padding:0 10px;border:1px solid var(--line,#E5E5E5);border-radius:6px;background:#fff;color:#555;font-size:13px;cursor:pointer;transition:all .15s;display:flex;align-items:center;justify-content:center}}
.rpt-page-btn:hover:not([disabled]){{border-color:var(--ink,#0C0C0C);color:var(--ink,#0C0C0C)}}
.rpt-page-btn.active{{background:var(--ink,#0C0C0C);border-color:var(--ink,#0C0C0C);color:#fff;font-weight:600}}
.rpt-page-btn[disabled]{{opacity:.35;cursor:not-allowed}}
.rpt-page-info{{font-size:12px;color:#888;padding:0 8px}}

/* ── 空状态 ── */
.rpt-empty{{grid-column:1/-1;text-align:center;padding:80px 40px}}
.rpt-empty-icon{{width:56px;height:56px;margin:0 auto 16px;opacity:.2}}
.rpt-empty-title{{font-size:16px;font-weight:600;color:#555;margin-bottom:6px}}
.rpt-empty-desc{{font-size:13px;color:#888}}
.rpt-empty-cta{{display:inline-flex;align-items:center;gap:6px;margin-top:20px;padding:9px 18px;background:var(--ink,#0C0C0C);color:#fff;border-radius:7px;font-size:13px;font-weight:600;text-decoration:none;transition:opacity .15s}}
.rpt-empty-cta:hover{{opacity:.85}}
</style>

<script>
const _AGENT_NAMES={agent_names_js};
const _AGENT_ICONS={agent_icons_js};
const _AGENT_CAT_CLASS={agent_cat_class_js};
const _AGENT_CATS={agent_categories_js};

// 颜色映射（与 agents.html 一致）
const _CAT_COLORS={{
  'cat-supply':'#15803d','cat-ad':'#7c3aed','cat-voc':'#0891b2',
  'cat-attribution':'#b45309','cat-competitor':'#be185d','cat-listing':'#0369a1',
  'cat-pricing':'#c2410c','cat-risk':'#b91c1c','cat-analytics':'#1d4ed8',
  'cat-content':'#6d28d9','cat-tag':'#0f766e','cat-cs':'#0891b2',
  'cat-selection':'#059669',
}};
function _catColor(id){{return _CAT_COLORS[_AGENT_CAT_CLASS[id]]||'#555';}}

let _activeFilter='all';
let _currentRptIdx=-1;
const _FEISHU_HOOK_RPT=typeof _FEISHU_HOOK!=='undefined'?_FEISHU_HOOK:'';

function _initSeeds(){{
  try{{
    var ex=JSON.parse(localStorage.getItem('agentReports')||'[]');
    if(ex.length===0){{
      var rootPrefix=window.location.pathname.includes('/skills/')||window.location.pathname.includes('/domains/')||window.location.pathname.includes('/playbooks/')||window.location.pathname.includes('/solutions/')?'../':'';
      fetch(rootPrefix+'assets/seed_reports.json').then(function(r){{return r.json();}}).then(function(seeds){{
        localStorage.setItem('agentReports',JSON.stringify(seeds));
        _renderR();
      }}).catch(function(){{}});
    }}
  }}catch(e){{}}
}}

function _loadR(){{try{{return JSON.parse(localStorage.getItem('agentReports')||'[]')}}catch(e){{return[]}}}}

function setAgentFilter(id){{
  _activeFilter=id;
  document.querySelectorAll('.rpt-filter').forEach(b=>b.classList.toggle('active',b.dataset.agent===id));
  const list=document.getElementById('rpt-list');
  list.dataset.page='1';
  _renderR();
}}

function _extractKPIs(result){{
  const kpis=[];
  const pats=[
    {{rx:/可售天数[：:]\\s*([\\d.]+)\\s*天/,label:'可售天数',suffix:'天',warnBelow:14}},
    {{rx:/净利润率[：:]\\s*([\\d.]+)%/,label:'净利润率',suffix:'%',warnBelow:8}},
    {{rx:/ACoS[：:]\\s*([\\d.]+)%/,label:'ACoS',suffix:'%',warnAbove:25}},
    {{rx:/ROAS[：:]\\s*([\\d.]+)/,label:'ROAS'}},
    {{rx:/机会评分[：:]\\s*(\\d+)\\/100/,label:'机会评分',suffix:'/100',warnBelow:60}},
    {{rx:/综合评分[：:]\\s*(\\d+)\\/100/,label:'综合评分',suffix:'/100',warnBelow:60}},
    {{rx:/账号健康[^:：]*[：:]\\s*(\\d+)/,label:'账号健康',suffix:'分',warnBelow:60}},
    {{rx:/退货率[：:]\\s*([\\d.]+)%/,label:'退货率',suffix:'%',warnAbove:15}},
    {{rx:/净利润[：:]\\s*\\$?([\\d,.]+)/,label:'净利润',prefix:'$'}},
  ];
  for(const p of pats){{
    const m=(result||'').match(p.rx);
    if(m){{
      const n=parseFloat(m[1].replace(/,/g,''));
      const w=(p.warnBelow!=null&&n<p.warnBelow)||(p.warnAbove!=null&&n>p.warnAbove);
      kpis.push({{label:p.label,value:(p.prefix||'')+m[1]+(p.suffix||''),warn:w}});
      if(kpis.length>=2)break;
    }}
  }}
  return kpis;
}}

function _renderReport(raw){{
  if(!raw)return'';
  const esc=s=>s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  const lines=raw.split(String.fromCharCode(10));
  let html='';
  for(let i=0;i<lines.length;i++){{
    const l=lines[i],trim=l.trim();
    if(!trim){{html+='<div class="rr-spacer"></div>';continue;}}
    if(/^━━.+━━$/.test(trim)){{
      const title=trim.replace(/^━+\\s*/,'').replace(/\\s*━+$/,'');
      html+=`<div class="rr-section"><span class="rr-section-dot"></span>${{esc(title)}}</div>`;continue;
    }}
    if(/^\\[!\\]|^\\[WARN\\]/.test(trim)){{
      html+=`<div class="rr-warn">⚠ ${{esc(trim.replace(/^\\[!\\]\\s*|\\[WARN\\]\\s*/,''))}}</div>`;continue;
    }}
    if(/^\\[OK\\]/.test(trim)){{
      html+=`<div class="rr-ok">✓ ${{esc(trim.replace(/^\\[OK\\]\\s*/,''))}}</div>`;continue;
    }}
    if(/^\\[>\\]/.test(trim)){{
      html+=`<div class="rr-action">→ ${{esc(trim.replace(/^\\[>\\]\\s*/,''))}}</div>`;continue;
    }}
    if(/^[├└]─/.test(trim)){{
      const msg=trim.replace(/^[├└]─+\\s*/,'');
      const vm=msg.match(/^(.+?):\\s*(.+)$/);
      if(vm){{
        const isNum=/[\\$\\d,%\\+\\-]+/.test(vm[2]);
        const val=isNum?`<span class="rr-val">${{esc(vm[2])}}</span>`:esc(vm[2]);
        html+=`<div class="rr-row"><span class="rr-row-key">${{esc(vm[1])}}</span><span class="rr-row-sep">:</span>${{val}}</div>`;
      }}else{{html+=`<div class="rr-row-plain">${{esc(msg)}}</div>`;}}
      continue;
    }}
    if(/^P[0-3][（（(]/.test(trim)||/^P[0-3]:/.test(trim)){{
      const p=trim.match(/^(P\\d)/)?.[1]||'P1';
      const pc=p==='P0'?'rr-p0':p==='P1'?'rr-p1':'rr-p2';
      html+=`<div class="rr-priority"><span class="${{pc}}">${{p}}</span>${{esc(trim.replace(/^P\\d[^\\s]*/,'').trim())}}</div>`;continue;
    }}
    const kv=trim.match(/^(.{{2,20}})[：:]\\s*(.{{1,200}})$/);
    if(kv&&!trim.startsWith('http')){{
      const numRaw=kv[2],hasNum=/[\\$\\d,%]+/.test(numRaw)&&numRaw.length<60;
      const val=hasNum?`<span class="rr-val">${{esc(numRaw)}}</span>`:esc(numRaw);
      html+=`<div class="rr-kv"><span class="rr-kv-key">${{esc(kv[1])}}</span><span class="rr-kv-sep">:</span>${{val}}</div>`;continue;
    }}
    let styled=esc(trim).replace(/(\\$[\\d,]+(?:\\.\\d+)?)/g,'<span class="rr-num">$1</span>').replace(/([\\d.]+%)/g,'<span class="rr-pct">$1</span>');
    html+=`<div class="rr-line">${{styled}}</div>`;
  }}
  return html;
}}

function _fmtTs(ts){{
  if(!ts)return'—';
  const d=new Date(ts.replace(' ','T'));
  if(isNaN(d))return ts;
  const today=new Date();
  const isToday=d.toDateString()===today.toDateString();
  if(isToday)return'今天 '+d.toLocaleTimeString('zh-CN',{{hour:'2-digit',minute:'2-digit'}});
  return d.toLocaleDateString('zh-CN',{{month:'2-digit',day:'2-digit'}})+' '+d.toLocaleTimeString('zh-CN',{{hour:'2-digit',minute:'2-digit'}});
}}

function _renderSummaryCard(r,globalIdx){{
  const name=_AGENT_NAMES[r.id]||r.name||r.id;
  const catClass=_AGENT_CAT_CLASS[r.id]||'';
  const color=_catColor(r.id);
  const icon=_AGENT_ICONS[r.id]||'';
  const ts=_fmtTs(r.ts);
  const category=_AGENT_CATS[r.id]||'';
  const kpis=_extractKPIs(r.result||'');

  // 输入参数摘要（最多2个）
  const inputEntries=Object.entries(r.inputs||{{}}).slice(0,2);
  const chips=inputEntries.map(([k,v])=>
    `<span class="rpt-card-input-chip">
      <span class="rpt-card-input-key">${{k}}</span>
      <span class="rpt-card-input-val" title="${{v.replace(/"/g,"'")}}">
        ${{v.length>25?v.slice(0,23)+'…':v}}
      </span>
    </span>`
  ).join('');

  // KPI 亮点
  const kpiHtml=kpis.length?`<div class="rpt-card-kpi-row">${{
    kpis.map(k=>`<div class="rpt-card-kpi">
      <div class="rpt-card-kpi-label">${{k.label}}</div>
      <div class="rpt-card-kpi-val ${{k.warn?'warn':'ok'}}">${{k.value}}</div>
    </div>`).join('')
  }}</div>`:'';

  // 结果文字预览（去掉格式符）
  const preview=(r.result||'').replace(/━+|\\[OK\\]|\\[!\\]|\\[>\\]/g,'').replace(/\\n+/g,' ').trim().slice(0,120);

  const seededBadge=r.seeded?'<span class="rpt-card-seeded">示例</span>':'';

  return `<div class="rpt-card" onclick="openRptDetail(${{globalIdx}})">
    <div class="rpt-card-top">
      <div class="rpt-card-icon-wrap" style="background:${{color}}">${{icon}}</div>
      <div class="rpt-card-info">
        <div class="rpt-card-agent-name">${{name}}</div>
        <div class="rpt-card-ts">${{ts}}</div>
      </div>
    </div>
    <div class="rpt-card-mid">
      ${{chips?`<div class="rpt-card-inputs-preview">${{chips}}</div>`:''}}
      ${{kpiHtml||`<div class="rpt-card-result-preview">${{preview}}</div>`}}
    </div>
    <div class="rpt-card-footer">
      ${{category?`<span class="rpt-card-cat-badge">${{category}}</span>`:''}}
      ${{seededBadge}}
      <span class="rpt-card-view-btn">查看详情
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"/></svg>
      </span>
    </div>
  </div>`;
}}

function _updateSummary(reports){{
  const today=new Date().toDateString();
  document.getElementById('rpt-total').textContent=reports.length;
  document.getElementById('rpt-agents').textContent=new Set(reports.map(r=>r.id)).size;
  document.getElementById('rpt-today').textContent=reports.filter(r=>r.ts&&new Date(r.ts.replace(' ','T')).toDateString()===today).length;
  const last=reports[reports.length-1];
  document.getElementById('rpt-latest').textContent=last?_fmtTs(last.ts):'—';
}}

function _renderR(){{
  const reports=_loadR();
  _updateSummary(reports);
  const filtered=_activeFilter==='all'?reports:reports.filter(r=>r.id===_activeFilter);
  const list=document.getElementById('rpt-list');
  if(!filtered.length){{
    list.innerHTML=`<div class="rpt-empty">
      <svg class="rpt-empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
      <div class="rpt-empty-title">暂无报告记录</div>
      <div class="rpt-empty-desc">前往智能体广场运行分析，报告将自动保存到这里</div>
      <a href="agents.html" class="rpt-empty-cta">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>
        前往智能体广场
      </a>
    </div>`;
    return;
  }}
  const PAGE_SIZE=9;
  const reversed=[...filtered].reverse();
  const totalPages=Math.ceil(reversed.length/PAGE_SIZE);
  let curPage=parseInt(list.dataset.page||'1');
  if(curPage<1)curPage=1;
  if(curPage>totalPages)curPage=totalPages;
  list.dataset.page=curPage;
  const start=(curPage-1)*PAGE_SIZE;
  const pageItems=reversed.slice(start,start+PAGE_SIZE);

  // 计算 globalIdx（在完整 reports 数组中的真实索引）
  const cards=pageItems.map((r,i)=>{{
    const fi=filtered.length-1-start-i;
    const gi=reports.indexOf(filtered[fi]);
    return _renderSummaryCard(r,gi);
  }}).join('');

  const pager=totalPages<=1?'':(() => {{
    let btns=`<button class="rpt-page-btn" onclick="_goPage(${{curPage-1}})" ${{curPage<=1?'disabled':''}}>‹</button>`;
    for(let p=1;p<=totalPages;p++){{
      if(totalPages>7&&p>2&&p<totalPages-1&&Math.abs(p-curPage)>1){{
        if(p===3||p===totalPages-2)btns+='<span class="rpt-page-info">…</span>';
        continue;
      }}
      btns+=`<button class="rpt-page-btn ${{p===curPage?'active':''}}" onclick="_goPage(${{p}})">${{p}}</button>`;
    }}
    btns+=`<button class="rpt-page-btn" onclick="_goPage(${{curPage+1}})" ${{curPage>=totalPages?'disabled':''}}>›</button>`;
    btns+=`<span class="rpt-page-info">${{start+1}}-${{Math.min(start+PAGE_SIZE,filtered.length)}}/${{filtered.length}}条</span>`;
    return`<div class="rpt-pagination">${{btns}}</div>`;
  }})();

  list.innerHTML=cards+pager;
}}

function _goPage(p){{
  const list=document.getElementById('rpt-list');
  list.dataset.page=p;
  _renderR();
  list.scrollIntoView({{behavior:'smooth',block:'start'}});
}}

/* ── 详情 Modal ── */
function openRptDetail(idx){{
  const r=_loadR()[idx];
  if(!r)return;
  _currentRptIdx=idx;
  const name=_AGENT_NAMES[r.id]||r.name||r.id;
  const color=_catColor(r.id);
  const icon=_AGENT_ICONS[r.id]||'';

  document.getElementById('rpt-detail-icon').innerHTML=icon;
  document.getElementById('rpt-detail-icon').style.background=color;
  document.getElementById('rpt-detail-agent-name').textContent=name;
  document.getElementById('rpt-detail-ts').textContent=_fmtTs(r.ts);

  // 输入参数
  const inputChips=Object.entries(r.inputs||{{}}).map(([k,v])=>
    `<span class="rpt-detail-input-chip">
      <span class="rpt-detail-input-key">${{k}}:</span>
      <span class="rpt-detail-input-val" title="${{v.replace(/"/g,"'")}}">${{v.length>40?v.slice(0,38)+'…':v}}</span>
    </span>`
  ).join('');
  const inputsEl=document.getElementById('rpt-detail-inputs');
  inputsEl.style.display=inputChips?'flex':'none';
  inputsEl.innerHTML=inputChips;

  document.getElementById('rpt-detail-body').innerHTML=`<div class="rr-body">${{_renderReport(r.result||'')}}</div>`;

  const overlay=document.getElementById('rpt-detail-overlay');
  overlay.style.display='flex';
  document.body.style.overflow='hidden';
}}

function closeRptDetail(){{
  document.getElementById('rpt-detail-overlay').style.display='none';
  document.body.style.overflow='';
  _currentRptIdx=-1;
}}

function copyCurrentRpt(){{
  if(_currentRptIdx<0)return;
  const r=_loadR()[_currentRptIdx];
  if(!r)return;
  const name=_AGENT_NAMES[r.id]||r.name||r.id;
  navigator.clipboard.writeText(`[${{name}}] ${{r.ts||''}}\\\n\\\n输入:\\\n${{JSON.stringify(r.inputs,null,2)}}\\\n\\\n结果:\\\n${{r.result||''}}`).then(()=>{{
    const btn=document.querySelector('.rpt-detail-btn:nth-child(2)');
    if(btn){{const orig=btn.innerHTML;btn.innerHTML='✓ 已复制';setTimeout(()=>btn.innerHTML=orig,1800);}}
  }});
}}

function deleteCurrentRpt(){{
  if(_currentRptIdx<0||!confirm('确认删除这条报告记录？'))return;
  const rs=_loadR();
  rs.splice(_currentRptIdx,1);
  localStorage.setItem('agentReports',JSON.stringify(rs));
  closeRptDetail();
  _renderR();
}}

function pushDetailToFeishu(){{
  if(_currentRptIdx<0)return;
  const r=_loadR()[_currentRptIdx];
  if(!r||!_FEISHU_HOOK_RPT){{alert('飞书推送未配置');return;}}
  const name=_AGENT_NAMES[r.id]||r.name||r.id;
  const btn=document.getElementById('rpt-detail-feishu-btn');
  if(btn){{btn.textContent='推送中…';btn.disabled=true;}}
  const inpLines=Object.entries(r.inputs||{{}}).map(([k,v])=>k+': '+v).join(String.fromCharCode(10));
  const resultText=(r.result||'').replace(/\\*\\*/g,'').replace(/#+\\s/g,'').trim().slice(0,1800);
  const body=JSON.stringify({{
    msg_type:'interactive',
    card:{{
      header:{{title:{{tag:'plain_text',content:'📊 '+name+' — 报告推送'}},template:'green'}},
      elements:[
        {{tag:'div',text:{{tag:'plain_text',content:'时间: '+r.ts+(inpLines?'\\\n\\\n输入:\\\n'+inpLines:'')}}}},
        {{tag:'hr'}},
        {{tag:'div',text:{{tag:'plain_text',content:resultText}}}}
      ]
    }}
  }});
  fetch(_FEISHU_HOOK_RPT,{{method:'POST',headers:{{'Content-Type':'application/json'}},body}})
    .then(()=>{{if(btn){{btn.innerHTML='✓ 已推送';setTimeout(()=>{{btn.innerHTML='推送飞书';btn.disabled=false;}},2000);}};}})
    .catch(()=>{{if(btn){{btn.textContent='推送失败';btn.disabled=false;}};}});
}}

// ESC 关闭
document.addEventListener('keydown',e=>{{if(e.key==='Escape')closeRptDetail();}});
document.getElementById('rpt-detail-overlay').addEventListener('click',e=>{{
  if(e.target===document.getElementById('rpt-detail-overlay'))closeRptDetail();
}});

function exportReports(){{
  const rs=_loadR();if(!rs.length){{alert('暂无报告记录');return;}}
  const txt=rs.map((r,i)=>`=== 报告 #${{i+1}} | ${{r.name||r.id}} | ${{r.ts||''}} ===\\\n输入:\\\n${{JSON.stringify(r.inputs,null,2)}}\\\n\\\n结果:\\\n${{r.result||''}}\\\n`).join('\\\n');
  const a=document.createElement('a');a.href='data:text/plain;charset=utf-8,'+encodeURIComponent(txt);a.download='agent-reports-'+new Date().toISOString().slice(0,10)+'.txt';a.click();
}}

function clearReports(){{
  if(!confirm('确认清空全部运行记录？此操作不可撤销。'))return;
  localStorage.removeItem('agentReports');_renderR();
}}

function _loadRemoteReports(){{
  (async function(){{
    try{{
      if(typeof _sessionKey!=='function')return;
      const remote=await fetch('/api/reports?session_key='+_sessionKey()+'&limit=20').then(r=>r.json());
      if(remote&&remote.length){{
        const local=_loadR();
        const seen=new Set(local.map(r=>r.id+'|'+(r.ts||'')));
        const merged=remote
          .filter(r=>!seen.has((r.agent_id||'')+'|'+(r.created_at||'')))
          .map(r=>{{return{{id:r.agent_id,name:r.agent_name,result:r.result,ts:r.created_at,inputs:JSON.parse(r.inputs||'{{}}')}};}})
          .concat(local).slice(0,50);
        if(remote.length&&!local.length){{localStorage.setItem('agentReports',JSON.stringify(merged));_renderR();}}
      }}
    }}catch(e){{}}
  }})();
}}

document.addEventListener('DOMContentLoaded',function(){{_initSeeds();_renderR();_loadRemoteReports();}});
window.addEventListener('storage',e=>{{if(e.key==='agentReports')_renderR();}});
</script>
"""
    return _html_page("智能体报告", body, active_nav="agent-report")
