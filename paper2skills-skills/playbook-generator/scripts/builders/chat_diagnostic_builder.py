"""Chat and Diagnostic page builders for paper2skills Playbook."""
from __future__ import annotations


def render_diagnostic_page(skill_count: int, build_ts: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>业务诊断中心 — paper2skills</title>
<link rel="stylesheet" href="assets/style.css">
<style>
.diag-wrap{{display:flex;gap:0;min-height:calc(100vh - var(--topbar-height,52px));background:var(--bg,#F6F6F6)}}
.diag-left{{
  width:320px;min-width:280px;flex-shrink:0;
  background:var(--panel,#fff);
  border-right:1px solid var(--line,#e4e4e4);
  padding:0;
  display:flex;flex-direction:column;
  position:sticky;top:var(--topbar-height,52px);
  height:calc(100vh - var(--topbar-height,52px));
  overflow-y:auto;overflow-x:hidden;
}}
.diag-left::-webkit-scrollbar{{width:3px}}
.diag-left::-webkit-scrollbar-track{{background:transparent}}
.diag-left::-webkit-scrollbar-thumb{{background:var(--line-strong,#ccc);border-radius:2px}}
.diag-left-head{{
  padding:20px 16px 14px;
  border-bottom:1px solid var(--line,#e4e4e4);
  flex-shrink:0;
}}
.diag-left-eyebrow{{
  font-size:10px;font-weight:700;letter-spacing:.09em;
  text-transform:uppercase;color:var(--muted,#888);
  margin-bottom:5px;
}}
.diag-title{{font-size:15px;font-weight:700;color:var(--ink,#1A1A2E);letter-spacing:-.02em;line-height:1.3}}
.diag-sub{{font-size:12px;color:var(--muted,#888);line-height:1.5;margin-top:3px}}
.diag-search-wrap{{padding:12px 16px;border-bottom:1px solid var(--line,#e4e4e4);flex-shrink:0}}
.diag-input-row{{display:flex;gap:7px}}
.diag-input{{
  flex:1;padding:8px 12px;
  border:1.5px solid var(--line,#e4e4e4);
  border-radius:var(--r-md,6px);
  font-size:13px;font-family:var(--font);
  outline:none;color:var(--ink);
  background:var(--bg,#F6F6F6);
  transition:border-color var(--t),background var(--t);
}}
.diag-input:focus{{border-color:var(--accent,#B5323E);background:#fff;box-shadow:0 0 0 3px rgba(181,50,62,.08)}}
.diag-input::placeholder{{color:var(--muted,#888);font-size:12.5px}}
.diag-btn{{
  padding:8px 14px;
  background:var(--accent,#B5323E);color:#fff;
  border:none;border-radius:var(--r-md,6px);
  font-size:12.5px;font-weight:600;cursor:pointer;
  white-space:nowrap;font-family:var(--font);
  transition:background var(--t);flex-shrink:0;
}}
.diag-btn:hover{{background:var(--accent-dark,#8C2530)}}
.diag-section-label{{
  font-size:10px;font-weight:700;letter-spacing:.09em;
  text-transform:uppercase;color:var(--muted,#888);
  padding:12px 16px 6px;
  user-select:none;
}}
.diag-events{{display:flex;flex-direction:column;gap:1px;padding:0 8px 8px}}
.diag-event-btn{{
  display:flex;align-items:center;gap:10px;
  padding:9px 10px;
  background:transparent;
  border:none;border-radius:var(--r-md,6px);
  cursor:pointer;text-align:left;font-family:var(--font);
  transition:background var(--t),color var(--t);
  position:relative;width:100%;
}}
.diag-event-btn:hover{{background:var(--panel-2,#f3f3f3)}}
.diag-event-btn.active{{
  background:var(--accent-bg,#FDF0F1);
}}
.diag-event-btn.active::before{{
  content:'';position:absolute;left:0;top:6px;bottom:6px;
  width:2px;border-radius:0 2px 2px 0;
  background:var(--accent,#B5323E);
}}
.diag-event-info{{min-width:0;flex:1}}
.diag-event-name{{font-size:12.5px;font-weight:500;color:var(--ink,#1A1A2E);line-height:1.35}}
.diag-event-btn.active .diag-event-name{{font-weight:600;color:var(--accent,#B5323E)}}
.diag-event-sev{{
  font-size:10px;padding:1px 6px;border-radius:3px;
  display:inline-block;margin-top:3px;font-weight:600;
  letter-spacing:.02em;
}}
.sev-critical{{background:#fef2f2;color:#991b1b}}
.sev-high{{background:#fff7ed;color:#c2410c}}
.sev-medium{{background:#eff6ff;color:#1d4ed8}}
.sev-low{{background:#f0fdf4;color:#166534}}
.diag-right{{flex:1;padding:28px 28px;overflow-y:auto}}
.diag-empty{{display:flex;flex-direction:column;align-items:center;justify-content:center;height:300px;color:var(--muted,#888);text-align:center;gap:12px}}
.diag-empty-icon{{font-size:40px;opacity:.5}}
.diag-empty-text{{font-size:14px;line-height:1.6;max-width:300px}}
.diag-result{{display:none}}
.diag-result.visible{{display:block}}
.diag-result-header{{background:#fff;border:1px solid var(--line,#e5e7eb);border-radius:10px;padding:20px 22px;margin-bottom:16px}}
.diag-result-title{{display:flex;align-items:center;gap:10px;margin-bottom:8px}}
.diag-result-icon{{font-size:24px}}
.diag-result-name{{font-size:17px;font-weight:700;color:var(--ink)}}
.diag-result-summary{{font-size:13px;color:var(--muted);line-height:1.6}}
.diag-phases{{display:flex;flex-direction:column;gap:12px}}
.diag-phase{{background:#fff;border:1px solid var(--line,#e5e7eb);border-radius:10px;overflow:hidden}}
.diag-phase-header{{padding:12px 18px;font-size:13px;font-weight:700;display:flex;align-items:center;gap:8px;border-bottom:1px solid var(--line,#e5e7eb)}}
.diag-phase-diagnose .diag-phase-header{{background:#f0f9ff;color:#0369a1}}
.diag-phase-treat .diag-phase-header{{background:#fff7ed;color:#b45309}}
.diag-phase-prevent .diag-phase-header{{background:#f0fdf4;color:#166534}}
.diag-skill-list{{padding:8px 0}}
.diag-skill-item{{display:flex;align-items:flex-start;gap:10px;padding:9px 18px;border-bottom:1px solid #f3f4f6;transition:background .12s}}
.diag-skill-item:last-child{{border-bottom:none}}
.diag-skill-item:hover{{background:#f9fafb}}
.diag-skill-num{{font-size:11px;font-weight:700;color:#9ca3af;flex-shrink:0;padding-top:2px;min-width:16px}}
.diag-skill-body{{min-width:0}}
.diag-skill-link{{font-size:12.5px;font-weight:600;color:var(--accent,#B5323E);text-decoration:none;display:block}}
.diag-skill-link:hover{{text-decoration:underline}}
.diag-skill-role{{font-size:12px;color:var(--muted);line-height:1.5;margin-top:2px}}
.diag-skill-cond{{font-size:11px;background:#fef9c3;color:#854d0e;padding:1px 6px;border-radius:4px;display:inline-block;margin-top:3px}}
.diag-related{{margin-top:16px;background:#fff;border:1px solid var(--line);border-radius:10px;padding:14px 18px}}
.diag-related-title{{font-size:12px;font-weight:700;color:var(--muted);margin-bottom:8px;text-transform:uppercase;letter-spacing:.5px}}
.diag-related-links{{display:flex;flex-wrap:wrap;gap:6px}}
.diag-related-link{{font-size:12px;padding:4px 10px;background:var(--bg);border:1px solid var(--line);border-radius:20px;text-decoration:none;color:var(--ink);transition:all .12s}}
.diag-related-link:hover{{border-color:var(--accent);color:var(--accent)}}
.path-finder{{
  margin:8px 8px;
  border:1px solid var(--line,#e4e4e4);
  border-radius:var(--r-lg,8px);overflow:hidden;
  flex-shrink:0;
}}
.path-finder-header{{
  padding:10px 14px;font-size:12.5px;font-weight:600;
  color:var(--ink-2);cursor:pointer;
  display:flex;align-items:center;justify-content:space-between;
  background:var(--panel-2,#f3f3f3);
  transition:background var(--t);user-select:none;
}}
.path-finder-header:hover{{background:var(--panel-3,#ececec)}}
.path-finder-arrow{{font-size:10px;color:var(--muted);transition:transform var(--t)}}
.path-finder-header.open .path-finder-arrow{{transform:rotate(90deg)}}
.path-finder-body{{padding:12px 14px;display:none;border-top:1px solid var(--line)}}
.path-finder-body.open{{display:block}}
.path-select{{
  width:100%;padding:7px 10px;
  border:1.5px solid var(--line);
  border-radius:var(--r-md,6px);
  font-size:12.5px;font-family:var(--font);
  margin-bottom:7px;outline:none;
  background:var(--bg);color:var(--ink);
  transition:border-color var(--t);
}}
.path-select:focus{{border-color:var(--accent)}}
.path-run-btn{{width:100%;padding:8px;background:var(--accent);color:#fff;border:none;border-radius:var(--r-md,6px);font-size:12.5px;font-weight:600;cursor:pointer;font-family:var(--font);transition:background var(--t)}}
.path-run-btn:hover{{background:var(--accent-dark)}}
.path-result{{margin-top:14px}}
.path-step{{display:flex;align-items:flex-start;gap:8px;padding:8px 0;border-bottom:1px solid #f3f4f6}}
.path-step:last-child{{border-bottom:none}}
.path-step-num{{font-size:11px;font-weight:700;color:#fff;background:var(--accent);border-radius:50%;width:20px;height:20px;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px}}
.path-step-link{{font-size:12.5px;font-weight:600;color:var(--accent);text-decoration:none;display:block}}
.path-step-link:hover{{text-decoration:underline}}
.path-step-domain{{font-size:11px;color:var(--muted);margin-top:1px}}
.path-edge-type{{font-size:10px;padding:1px 5px;border-radius:3px;margin-left:6px;vertical-align:middle}}
.pet-prerequisite{{background:#f0f9ff;color:#0369a1}}.pet-extension{{background:#f0fdf4;color:#166534}}.pet-combinable{{background:#fef3c7;color:#92400e}}
@media(max-width:768px){{.diag-wrap{{flex-direction:column}}.diag-left{{width:100%;position:static;height:auto;border-right:none;border-bottom:1px solid var(--line)}}.diag-right{{padding:16px}}}}
</style>
</head>
<body>
<header class="topbar">
  <button class="hamburger" id="hamburger-diag" aria-label="菜单" aria-expanded="false">
    <span></span><span></span><span></span>
  </button>
  <a class="brand" href="index.html">
    <span class="brand-icon">P</span>
    <span class="brand-name">paper2skills<span class="brand-tag">Playbook</span></span>
  </a>
  <div class="topbar-right">
    <input id="global-search-diag" placeholder="搜索技能 / 场景…" autocomplete="off"
      style="width:min(220px,18vw);padding:6px 12px 6px 30px;border-radius:var(--r-sm);border:1px solid #2E2E2E;background:#1A1A1A url('data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%2212%22 height=%2212%22 viewBox=%220 0 24 24%22 fill=%22none%22 stroke=%22%23555%22 stroke-width=%222%22%3E%3Ccircle cx=%2211%22 cy=%2211%22 r=%228%22/%3E%3Cpath d=%22m21 21-4.35-4.35%22/%3E%3C/svg%3E') no-repeat 10px center;color:#CCCCCC;font-size:12.5px;font-family:var(--font)">
    <span class="topbar-stat">{skill_count} Skills</span>
    <a href="pricing.html" class="topbar-cta">升级 Pro →</a>
    <div id="p2s-auth-widget" style="margin-left:4px;display:flex;align-items:center;gap:8px">
      <span id="p2s-user-avatar-d" style="width:22px;height:22px;border-radius:999px;display:none;align-items:center;justify-content:center;background:#111827;color:#fff;font-size:11px;font-weight:700;flex:0 0 auto"></span>
      <a id="p2s-login-btn-d" href="/auth/login" style="font-size:11px;padding:4px 10px;background:#B5323E;color:#fff;border-radius:5px;text-decoration:none;white-space:nowrap">飞书登录</a>
    </div>
  </div>
</header>
<div class="diag-wrap">
  <aside class="diag-left">
    <div class="diag-left-head">
      <div class="diag-left-eyebrow">paper2skills · 诊断中心</div>
      <div class="diag-title">症状 → Skill 链</div>
      <div class="diag-sub">描述业务问题，匹配「诊断→处置→预防」三层技能</div>
    </div>
    <div class="diag-search-wrap">
      <div class="diag-input-row">
        <input class="diag-input" id="diag-input" placeholder="例：ASIN 流量下降、广告 ACoS 过高…" />
        <button class="diag-btn" onclick="runDiag()">诊断</button>
      </div>
    </div>
    <div class="diag-section-label">风险症状</div>
    <div class="diag-events" id="diag-events"></div>
    <div class="path-finder">
      <div class="path-finder-header" id="pf-header" onclick="togglePathFinder()">
        <span>Skill 路径规划</span>
        <span class="path-finder-arrow" id="pf-arrow">▸</span>
      </div>
      <div class="path-finder-body" id="path-finder-body">
        <select class="path-select" id="pf-from" title="起点 Skill"></select>
        <select class="path-select" id="pf-to" title="目标 Skill"></select>
        <button class="path-run-btn" onclick="runPathFinder()">查找路径 →</button>
      </div>
    </div>
  </aside>
  <main class="diag-right" id="diag-right">
    <div class="diag-empty" id="diag-empty">
      <div class="diag-empty-icon">—</div>
      <div class="diag-empty-title" style="font-size:16px;font-weight:700;color:var(--ink);margin-bottom:6px">症状 → 诊断链 → 行动方案</div>
      <div class="diag-empty-text">点击左侧症状，获取「诊断→处置→预防」三层 Skill 链</div>
      <div style="margin-top:20px;display:grid;grid-template-columns:1fr 1fr;gap:8px;max-width:400px;text-align:left">
        <div style="padding:10px 12px;background:var(--bg);border:1px solid var(--line);border-radius:8px;font-size:12px;color:var(--ink-2)">
          <div style="font-weight:700;margin-bottom:3px">诊断层</div>找出根因的算法技能
        </div>
        <div style="padding:10px 12px;background:var(--bg);border:1px solid var(--line);border-radius:8px;font-size:12px;color:var(--ink-2)">
          <div style="font-weight:700;margin-bottom:3px">处置层</div>立即执行的行动 Skill
        </div>
        <div style="padding:10px 12px;background:var(--bg);border:1px solid var(--line);border-radius:8px;font-size:12px;color:var(--ink-2)">
          <div style="font-weight:700;margin-bottom:3px">预防层</div>长效防范的系统 Skill
        </div>
        <div style="padding:10px 12px;background:var(--bg);border:1px solid var(--line);border-radius:8px;font-size:12px;color:var(--ink-2)">
          <div style="font-weight:700;margin-bottom:3px">手册</div>配套可执行场景手册
        </div>
      </div>
    </div>
    <div class="diag-result" id="diag-result"></div>
  </main>
</div>
<script src="assets/risk-events.js?v={build_ts}"></script>
<script>
(function(){{
  const SEV_CLASS={{'critical':'sev-critical','high':'sev-high','medium':'sev-medium','low':'sev-low'}};
  const SEV_LABEL={{'critical':'紧急','high':'高风险','medium':'中风险','low':'低风险'}};
  const PHASE_CFG={{
    diagnose:{{label:'第一步：诊断根因',cls:'diag-phase-diagnose'}},
    treat:{{label:'第二步：处置行动',cls:'diag-phase-treat'}},
    prevent:{{label:'第三步：长效预防',cls:'diag-phase-prevent'}}
  }};

  function initButtons(){{
    const events=(window.RISK_EVENTS||{{}}).events||[];
    const container=document.getElementById('diag-events');
    container.innerHTML='';
    events.forEach(ev=>{{
      const btn=document.createElement('button');
      btn.className='diag-event-btn';
      btn.dataset.id=ev.event_id;
      const sevCls=SEV_CLASS[ev.severity]||'sev-medium';
      const sevLabel=SEV_LABEL[ev.severity]||ev.severity;
      btn.innerHTML=`<div class="diag-event-info"><div class="diag-event-name">${{ev.event_name}}</div><span class="diag-event-sev ${{sevCls}}">${{sevLabel}}</span></div>`;
      btn.addEventListener('click',()=>showEvent(ev.event_id));
      container.appendChild(btn);
    }});
  }}

  function matchEvent(query){{
    const events=(window.RISK_EVENTS||{{}}).events||[];
    const q=query.toLowerCase();
    let best=null,top=0;
    events.forEach(ev=>{{
      let sc=0;
      (ev.symptom_keywords||[]).forEach(kw=>{{if(q.includes(kw.toLowerCase()))sc++;}});
      if(sc>top){{top=sc;best=ev;}}
    }});
    return top>0?best:null;
  }}

  function showEvent(eventId){{
    const events=(window.RISK_EVENTS||{{}}).events||[];
    const ev=events.find(e=>e.event_id===eventId);
    if(!ev)return;
    document.querySelectorAll('.diag-event-btn').forEach(b=>b.classList.remove('active'));
    const btn=document.querySelector(`.diag-event-btn[data-id="${{eventId}}"]`);
    if(btn)btn.classList.add('active');
    const sevCls=SEV_CLASS[ev.severity]||'sev-medium';
    const sevLabel=SEV_LABEL[ev.severity]||ev.severity;
    let html=`<div class="diag-result-header"><div class="diag-result-title"><span class="diag-result-name">${{ev.event_name}}</span><span class="diag-event-sev ${{sevCls}}" style="margin-left:8px">${{sevLabel}}</span></div><div class="diag-result-summary">${{ev.summary||''}}</div></div><div class="diag-phases">`;
    ['diagnose','treat','prevent'].forEach(phase=>{{
      const skills=(ev.phases||{{}})[phase]||[];
      if(!skills.length)return;
      const cfg=PHASE_CFG[phase];
      let items='';
      skills.forEach((sk,i)=>{{
        const cond=sk.condition?`<span class="diag-skill-cond">触发条件：${{sk.condition}}</span>`:'';
        const title=sk.title?` — ${{sk.title.split('—')[0].trim()}}`:'';
        items+=`<div class="diag-skill-item"><span class="diag-skill-num">${{i+1}}</span><div class="diag-skill-body"><a class="diag-skill-link" href="skills/${{sk.skill_id}}.html" target="_blank">${{sk.skill_id}}</a><div class="diag-skill-role">${{sk.role||''}}</div>${{cond}}</div></div>`;
      }});
      html+=`<div class="diag-phase ${{cfg.cls}}"><div class="diag-phase-header">${{cfg.label}}<span style="margin-left:auto;font-size:11px;font-weight:400;opacity:.7">${{skills.length}} 个 Skills</span></div><div class="diag-skill-list">${{items}}</div></div>`;
    }});
    html+='</div>';
    const pbs=(ev.related_playbooks||[]);
    if(pbs.length){{
      const links=pbs.map(id=>`<a class="diag-related-link" href="playbooks/${{id}}.html" target="_blank">${{id.replace('pb-','').replace(/-/g,' ')}}</a>`).join('');
      html+=`<div class="diag-related"><div class="diag-related-title">相关手册</div><div class="diag-related-links">${{links}}</div></div>`;
    }}
    document.getElementById('diag-empty').style.display='none';
    const res=document.getElementById('diag-result');
    res.innerHTML=html;
    res.className='diag-result visible';
  }}

  window.runDiag=function(){{
    const q=document.getElementById('diag-input').value.trim();
    if(!q)return;
    const ev=matchEvent(q);
    if(ev){{showEvent(ev.event_id);}}
    else{{
      document.getElementById('diag-empty').style.display='none';
      document.getElementById('diag-result').className='diag-result visible';
      document.getElementById('diag-result').innerHTML='<div class="diag-empty"><div class="diag-empty-icon">—</div><div class="diag-empty-text">未匹配到具体风险场景<br>请尝试点击左侧症状按钮，或前往 <a href="chat.html">AI对话</a> 获取帮助</div></div>';
    }}
  }};

  document.getElementById('diag-input').addEventListener('keydown',e=>{{if(e.key==='Enter')window.runDiag();}});
  if(window.RISK_EVENTS)initButtons();
  else window.addEventListener('load',initButtons);
}})();

(function(){{
  let gNodes={{}}, gAdj={{}};

  function loadGraph(cb){{
    if(Object.keys(gNodes).length){{cb();return;}}
    fetch('assets/graph-data.json').then(r=>r.json()).then(d=>{{
      d.nodes.forEach(n=>{{gNodes[n.id]={{id:n.id,domain:n.domain,title:n.title}};gAdj[n.id]=[];}});
      d.links.forEach(l=>{{if(gAdj[l.source])gAdj[l.source].push({{to:l.target,type:l.type}});}});
      const from=document.getElementById('pf-from'),to=document.getElementById('pf-to');
      const sorted=d.nodes.slice().sort((a,b)=>a.id.localeCompare(b.id));
      sorted.forEach(n=>{{
        const o1=document.createElement('option');o1.value=n.id;o1.textContent=n.id;from.appendChild(o1);
        const o2=document.createElement('option');o2.value=n.id;o2.textContent=n.id;to.appendChild(o2);
      }});
      cb();
    }}).catch(()=>{{}});
  }}

  function bfs(startId,endId){{
    if(startId===endId)return [{{id:startId,edgeType:''}}];
    const visited={{[startId]:true}},queue=[{{id:startId,path:[{{id:startId,edgeType:''}}]}}];
    while(queue.length){{
      const {{id,path}}=queue.shift();
      for(const nb of(gAdj[id]||[])){{
        if(visited[nb.to])continue;
        visited[nb.to]=true;
        const np=[...path,{{id:nb.to,edgeType:nb.type}}];
        if(nb.to===endId)return np;
        if(np.length<7)queue.push({{id:nb.to,path:np}});
      }}
    }}
    return null;
  }}

   window.togglePathFinder=function(){{
    const body=document.getElementById('path-finder-body');
    const header=document.getElementById('pf-header');
    const arrow=document.getElementById('pf-arrow');
    const open=body.classList.toggle('open');
    if(header)header.classList.toggle('open',open);
    arrow.textContent=open?'▾':'▸';
    if(open)loadGraph(()=>{{}});
  }};

  window.runPathFinder=function(){{
    const fromId=document.getElementById('pf-from').value;
    const toId=document.getElementById('pf-to').value;
    if(!fromId||!toId)return;
    loadGraph(()=>{{
      const path=bfs(fromId,toId);
      const empty=document.getElementById('diag-empty');
      const res=document.getElementById('diag-result');
      empty.style.display='none';
      if(!path){{
        res.className='diag-result visible';
        res.innerHTML=`<div class="diag-result-header"><div class="diag-result-name">未找到路径</div><div class="diag-result-summary">从 ${{fromId}} 到 ${{toId}} 在当前图谱中无可达路径（跳数≤6）。</div></div>`;
        return;
      }}
      const edgeLabel={{'prerequisite':'前置','extension':'延伸','combinable':'可组合'}};
      const edgeCls={{'prerequisite':'pet-prerequisite','extension':'pet-extension','combinable':'pet-combinable'}};
      let steps='';
      path.forEach((node,i)=>{{
        const n=gNodes[node.id]||{{}};
        const badge=node.edgeType?`<span class="path-edge-type ${{edgeCls[node.edgeType]||''}}">${{edgeLabel[node.edgeType]||node.edgeType}}</span>`:'';
        steps+=`<div class="path-step"><span class="path-step-num">${{i+1}}</span><div><a class="path-step-link" href="skills/${{node.id}}.html" target="_blank">${{node.id}}</a><div class="path-step-domain">${{n.domain||''}}${{badge}}</div></div></div>`;
      }});
      res.className='diag-result visible';
      res.innerHTML=`<div class="diag-result-header"><div class="diag-result-title"><span class="diag-result-name">Skill 路径：${{fromId}} → ${{toId}}</span></div><div class="diag-result-summary">${{path.length}} 步路径（${{path.length-1}} 条边）</div></div><div class="path-result">${{steps}}</div>`;
    }});
  }};
}})();
</script>
<script src="assets/search.js"></script>
<script src="assets/playbook-data.js"></script>
<script>
(function(){{
  const hbtn=document.getElementById('hamburger-diag');
  if(hbtn){{
    hbtn.addEventListener('click',function(){{
      const open=hbtn.getAttribute('aria-expanded')!=='true';
      hbtn.setAttribute('aria-expanded',open);
      hbtn.classList.toggle('open',open);
    }});
  }}
  fetch('/auth/me',{{credentials:'include',cache:'no-store'}}).then(function(r){{return r.json();}}).then(function(u){{
    if(u&&u.name){{
      var av=document.getElementById('p2s-user-avatar-d');
      var lb=document.getElementById('p2s-login-btn-d');
      if(av){{av.textContent=String(u.name).slice(0,1);av.style.display='inline-flex';}}
      if(lb){{lb.textContent=u.name;lb.href='/settings.html';lb.style.background='transparent';lb.style.color='#94a3b8';lb.style.border='1px solid rgba(255,255,255,.15)';}}
    }}
  }}).catch(function(){{}});
}})();
</script>
</body>
</html>"""


def render_chat_page(nav: str = "", skill_count: int = 0) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>AI 知识库对话 · paper2skills</title>
  <link rel="stylesheet" href="{nav}assets/style.css">
  <style>
    body {{ overflow: hidden; }}
    .chat-layout {{
      display: flex; height: 100vh; flex-direction: column;
    }}
    .chat-topbar {{
      display: flex; align-items: center; gap: 0;
      height: var(--topbar-height); flex-shrink: 0;
      padding: 0 20px;
      background: rgba(255,255,255,0.92);
      backdrop-filter: blur(12px) saturate(180%);
      -webkit-backdrop-filter: blur(12px) saturate(180%);
      border-bottom: 1px solid var(--nav-border);
      box-shadow: 0 1px 0 rgba(0,0,0,0.05);
    }}
    .chat-back {{
      display: flex; align-items: center; gap: 6px;
      color: var(--accent); text-decoration: none; font-size: 13px; font-weight: 500;
      padding: 6px 10px; border-radius: var(--r-md);
      transition: background var(--t);
      flex-shrink: 0;
    }}
    .chat-back:hover {{ background: var(--accent-light); text-decoration: none; }}
    .chat-title-area {{
      flex: 1; display: flex; align-items: center; justify-content: center;
      gap: 10px;
    }}
    .chat-title-icon {{
      font-size: 20px;
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .chat-title-text {{
      font-size: 15px; font-weight: 650; letter-spacing: -.02em; color: var(--ink);
    }}
    .chat-title-sub {{
      font-size: 11.5px; color: var(--muted); font-weight: 400;
      padding: 3px 9px; background: var(--panel-2); border-radius: var(--r-full);
      border: 1px solid var(--line);
    }}
    .chat-ctrl {{ flex-shrink: 0; display: flex; align-items: center; gap: 10px; }}
    .web-search-toggle {{
      display: inline-flex; align-items: center; gap: 5px;
      font-size: 12px; color: var(--muted); cursor: pointer;
      padding: 5px 10px; border-radius: var(--r-full);
      border: 1.5px solid var(--line); background: transparent;
      transition: all var(--t); user-select: none; flex-shrink: 0;
      font-family: var(--font); white-space: nowrap;
    }}
    .web-search-toggle.on {{
      color: var(--accent); border-color: var(--accent);
      background: var(--accent-light); font-weight: 600;
    }}
    .web-search-toggle:hover {{ border-color: var(--line-strong); color: var(--ink); }}
    .web-search-toggle.on:hover {{ border-color: var(--accent-dark); }}
    .web-search-toggle-icon {{ font-size: 13px; line-height: 1; }}
    .chat-body {{
      flex: 1; display: flex; flex-direction: column;
      max-width: 760px; width: 100%; margin: 0 auto;
      padding: 0 16px; overflow: hidden;
    }}
    .chat-messages {{
      flex: 1; overflow-y: auto; padding: 28px 0 12px;
      display: flex; flex-direction: column; gap: 20px;
    }}
    .chat-messages::-webkit-scrollbar {{ width: 4px; }}
    .chat-messages::-webkit-scrollbar-thumb {{ background: var(--line-strong); border-radius: 4px; }}
    .cmsg {{ display: flex; gap: 12px; align-items: flex-start; }}
    .cmsg-user {{ flex-direction: row-reverse; }}
    .cmsg-avatar {{
      width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0;
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%);
      color: #fff; display: flex; align-items: center; justify-content: center;
      font-size: 13px; font-weight: 700; margin-top: 2px;
    }}
    .cmsg-user .cmsg-avatar {{
      background: var(--panel-3); color: var(--muted); font-size: 11px;
    }}
    .cmsg-body {{ flex: 1; min-width: 0; }}
    .cmsg-name {{
      font-size: 11px; font-weight: 600; letter-spacing: .02em;
      text-transform: uppercase; color: var(--muted); margin-bottom: 5px;
    }}
    .cmsg-user .cmsg-name {{ text-align: right; }}
    .cmsg-bubble {{
      display: inline-block; max-width: 100%;
      padding: 12px 16px; border-radius: 4px 18px 18px 18px;
      background: var(--panel); border: 1px solid var(--line);
      font-size: 14.5px; line-height: 1.72; color: var(--ink);
      box-shadow: var(--shadow-xs);
    }}
    .cmsg-user .cmsg-bubble {{
      background: var(--accent); color: #fff; border-color: transparent;
      border-radius: 18px 4px 18px 18px; box-shadow: none;
    }}
    .cmsg-bubble strong {{ font-weight: 700; }}
    .cmsg-bubble code {{
      background: rgba(0,0,0,.06); padding: 2px 6px;
      border-radius: 5px; font-size: 13px; font-family: 'SF Mono', 'Menlo', monospace;
    }}
    .cmsg-bubble br {{ margin: 0; }}
    .cmsg-web-badge {{
      display: inline-flex; align-items: center; gap: 4px;
      font-size: 11px; color: var(--muted); margin-bottom: 6px;
      padding: 2px 8px; background: var(--panel-2); border-radius: var(--r-full);
      border: 1px solid var(--line);
    }}
    .cmsg-event-badge {{
      display: inline-flex; align-items: center; gap: 4px;
      font-size: 11px; color: #991b1b; margin-bottom: 6px;
      padding: 2px 8px; background: #fef2f2; border-radius: var(--r-full);
      border: 1px solid #fecaca;
    }}
    .cmsg-typing .cmsg-bubble::after {{
      content: ''; display: inline-block; width: 40px; height: 10px;
      background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 10'%3E%3Ccircle cx='5' cy='5' r='3' fill='%2386868b'%3E%3Canimate attributeName='opacity' values='1;0.2;1' dur='1s' begin='0s' repeatCount='indefinite'/%3E%3C/circle%3E%3Ccircle cx='20' cy='5' r='3' fill='%2386868b'%3E%3Canimate attributeName='opacity' values='1;0.2;1' dur='1s' begin='0.2s' repeatCount='indefinite'/%3E%3C/circle%3E%3Ccircle cx='35' cy='5' r='3' fill='%2386868b'%3E%3Canimate attributeName='opacity' values='1;0.2;1' dur='1s' begin='0.4s' repeatCount='indefinite'/%3E%3C/circle%3E%3C/svg%3E") no-repeat center;
      vertical-align: middle; margin-left: 4px;
    }}
    .chat-welcome {{
      text-align: center; padding: 40px 20px 20px; color: var(--muted);
    }}
    .chat-welcome-icon {{
      font-size: 48px; display: block; margin-bottom: 16px;
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .chat-welcome h2 {{
      font-size: 22px; font-weight: 700; letter-spacing: -.03em; color: var(--ink);
      margin: 0 0 8px; border: none; padding: 0;
    }}
    .chat-welcome p {{ font-size: 14.5px; color: var(--muted); margin: 0 0 24px; }}
    .chat-suggestions {{
      display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 8px;
    }}
    .chat-sug-btn {{
      padding: 8px 16px; border-radius: var(--r-full);
      border: 1.5px solid var(--line); background: var(--panel);
      font-size: 13px; color: var(--ink-2); cursor: pointer; font-family: var(--font);
      transition: border-color var(--t), background var(--t), color var(--t);
    }}
    .chat-sug-btn:hover {{
      border-color: var(--accent); color: var(--accent); background: var(--accent-light);
    }}
    .chat-input-area {{
      flex-shrink: 0; padding: 10px 0 18px;
      border-top: 1px solid var(--line);
    }}
    .chat-input-wrap {{
      display: flex; align-items: flex-end; gap: 8px;
      background: var(--panel); border: 1.5px solid var(--line);
      border-radius: 18px; padding: 8px 10px 8px 12px;
      transition: border-color var(--t), box-shadow var(--t);
    }}
    .chat-input-wrap:focus-within {{
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(194,91,110,.10);
    }}
    .chat-input-wrap .web-search-toggle {{
      align-self: flex-end; margin-bottom: 1px;
    }}
    .chat-textarea {{
      flex: 1; border: none; outline: none; resize: none;
      font-family: var(--font); font-size: 14.5px; line-height: 1.6;
      color: var(--ink); background: transparent;
      min-height: 24px; max-height: 160px;
      overflow-y: auto;
    }}
    .chat-textarea::placeholder {{ color: var(--muted); }}
    .chat-send-btn {{
      width: 36px; height: 36px; border-radius: 50%; flex-shrink: 0;
      background: var(--accent); border: none; cursor: pointer;
      display: flex; align-items: center; justify-content: center;
      color: #fff; font-size: 16px;
      transition: background var(--t), transform var(--t);
    }}
    .chat-send-btn:hover {{ background: var(--accent-dark); transform: scale(1.06); }}
    .chat-send-btn:disabled {{ opacity: 0.45; cursor: not-allowed; transform: none; }}
    .chat-hint {{
      text-align: center; font-size: 11.5px; color: var(--muted);
      margin-top: 8px;
    }}
    @media (max-width: 600px) {{
      .chat-title-sub {{ display: none; }}
      .chat-body {{ padding: 0 12px; }}
    }}
  </style>
</head>
<body>
  <div class="chat-layout">
    <header class="chat-topbar">
      <a class="chat-back" href="{nav}index.html">← 返回</a>
      <div class="chat-title-area">
        <span class="chat-title-text">AI 知识库对话</span>
        <span class="chat-title-sub">{skill_count} Skills · DeepSeek V3</span>
      </div>
      <div class="chat-ctrl">
        <select id="role-select" style="font-size:12px;padding:5px 9px;border:1.5px solid var(--line);border-radius:var(--r-full);background:transparent;color:var(--ink);cursor:pointer;font-family:var(--font);">
          <option value="ops">运营视角</option>
          <option value="analyst">数据分析师</option>
          <option value="ceo">CEO 战略</option>
        </select>
      </div>
    </header>

    <div class="chat-body">
      <div class="chat-messages" id="chat-messages">
        <div class="chat-welcome" id="chat-welcome">
          <h2>paper2skills 知识库助手</h2>
          <p>基于 {skill_count} 个从顶会论文萃取的跨境电商 AI 决策技能，为你提供专业问答</p>
          <div style="font-size:11px;color:var(--muted);font-weight:600;letter-spacing:.5px;text-transform:uppercase;margin-bottom:8px">试试问这些</div>
          <div class="chat-suggestions">
            <button class="chat-sug-btn">如何提升广告 ROI？</button>
            <button class="chat-sug-btn">大促备货如何预测需求？</button>
            <button class="chat-sug-btn">供应链 AI 有哪些关键技能？</button>
            <button class="chat-sug-btn">KOL 投放效果怎么归因？</button>
            <button class="chat-sug-btn">账号 ODR 异常，担心被封号</button>
            <button class="chat-sug-btn">ASIN 流量突然下降 30%</button>
            <button class="chat-sug-btn">产品被平台合规警告</button>
          </div>
        </div>
      </div>

      <div class="chat-input-area">
        <div class="chat-input-wrap">
          <button class="web-search-toggle" id="web-search-toggle" title="开启联网搜索">
            <span class="web-search-toggle-icon">联网</span>
            <span id="web-search-label">联网搜索</span>
          </button>
          <textarea class="chat-textarea" id="chat-input"
            placeholder="描述业务症状，如：ASIN 流量下降、库存积压、广告 ACOS 过高…"
            rows="1" autocomplete="off"></textarea>
          <button class="chat-send-btn" id="chat-send" title="发送 (Enter)">↑</button>
        </div>
        <p class="chat-hint">Enter 发送 · Shift+Enter 换行</p>
      </div>
    </div>
  </div>

  <script src="{nav}assets/playbook-data.js"></script>
  <script src="{nav}assets/risk-events.js"></script>
  <script src="{nav}assets/chat-page.js"></script>
</body>
</html>"""


def build_chat_page_js() -> str:
    return r"""
(function () {
  const msgsEl = document.getElementById('chat-messages');
  const welcome = document.getElementById('chat-welcome');
  const textarea = document.getElementById('chat-input');
  const sendBtn = document.getElementById('chat-send');
  const webToggle = document.getElementById('web-search-toggle');
  const webLabel = document.getElementById('web-search-label');

  let webSearchOn = false;
  const _HIST_KEY = 'p2s_chat_v1';

  function _loadH() {
    try { return JSON.parse(localStorage.getItem(_HIST_KEY) || '[]'); } catch (e) { return []; }
  }

  function _saveH() {
    try { localStorage.setItem(_HIST_KEY, JSON.stringify(history.slice(-20))); } catch (e) {}
  }

  let history = _loadH();

  window.clearHistory = function() {
    history = [];
    try { localStorage.removeItem(_HIST_KEY); } catch (e) {}
    if (msgsEl) { msgsEl.innerHTML = ''; }
    if (welcome) welcome.style.display = '';
  };

  webToggle.addEventListener('click', () => {
    webSearchOn = !webSearchOn;
    webToggle.classList.toggle('on', webSearchOn);
    webLabel.textContent = webSearchOn ? '已开启联网' : '联网搜索';
  });

  textarea.addEventListener('input', () => {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 160) + 'px';
  });

  textarea.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      doSend();
    }
  });

  sendBtn.addEventListener('click', doSend);

  if (history.length && welcome) {
    welcome.style.display = 'none';
    history.forEach(function (m) {
      if (m.role === 'user') addMsg(m.content, 'user');
      else if (m.role === 'assistant') addMsg(m.content, 'bot');
    });
  }

  document.querySelectorAll('.chat-sug-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      textarea.value = btn.textContent.trim();
      textarea.dispatchEvent(new Event('input'));
      doSend();
    });
  });

  function md(text) {
    return text
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/\*\*(.+?)\*\*/gs, '<strong>$1</strong>')
      .replace(/\*([^*\n]+)\*/g, '<em>$1</em>')
      .replace(/`([^`\n]+)`/g, '<code>$1</code>')
      .replace(/^#{1,3}\s+(.+)$/gm, '<strong style="font-size:15px">$1</strong>')
      .replace(/^[-•]\s+(.+)$/gm, '<span style="display:block;padding-left:14px;margin:2px 0">• $1</span>')
      .replace(/^\d+\.\s+(.+)$/gm, '<span style="display:block;padding-left:14px;margin:2px 0">$&</span>')
      .replace(/\n\n+/g, '<br><br>').replace(/\n/g, '<br>');
  }

  const _idx = [];
  let _built = false;

  function buildSkillIndex() {
    if (_built) return;
    const DATA = window.PLAYBOOK_DATA || {};
    (DATA.skills || []).forEach(s => {
      const t = [
        s.skill_id || '', s.title || '', s.problem_solved || '',
        s.algorithm_summary || '', s.biz_trigger || '', s.biz_outcome || '',
        (s.tags || []).join(' '), (s.topics || []).join(' ')
      ].join(' ').toLowerCase();
      _idx.push({ s, t });
    });
    _built = true;
  }

  let _skillIdx = null;
  async function _loadSkillIdx() {
    if (_skillIdx) return _skillIdx;
    try {
      _skillIdx = await fetch('/assets/skill-index.json').then(r => r.json());
    } catch(e) { _skillIdx = []; }
    return _skillIdx;
  }

  async function _retrieveSkills(query, topK) {
    topK = topK || 5;
    const idx = await _loadSkillIdx();
    if (!idx || !idx.length) return [];
    const tokens = query.toLowerCase().replace(/[^\u4e00-\u9fa5a-z0-9\s]/g,' ').split(/\s+/).filter(function(t){ return t.length > 1; });
    if (!tokens.length) return [];
    const scored = idx.map(function(s) {
      const text = (s.summary + ' ' + s.keywords.join(' ')).toLowerCase();
      const score = tokens.reduce(function(n, t) { return n + (text.includes(t) ? 1 : 0); }, 0);
      return { s: s, score: score };
    }).filter(function(x) { return x.score > 0; });
    scored.sort(function(a,b) { return b.score - a.score; });
    return scored.slice(0, topK).map(function(x) { return x.s; });
  }

  function searchSkills(query, k) {
    k = k || 8;
    buildSkillIndex();
    const words = query.toLowerCase().split(/\s+/).filter(w => w.length > 1);
    if (!words.length) return [];
    return _idx.map(item => {
      let sc = 0;
      words.forEach(w => {
        const tf = item.t.split(w).length - 1;
        if (tf > 0) sc += tf * (w.length > 3 ? 2 : 1);
      });
      return { skill: item.s, sc };
    }).filter(x => x.sc > 0).sort((a, b) => b.sc - a.sc).slice(0, k).map(x => x.skill);
  }

  function buildRAGContext(query) {
    const top = searchSkills(query, 10);
    if (!top.length) {
      return (window.PLAYBOOK_DATA && window.PLAYBOOK_DATA.skills || []).slice(0, 60).map(s =>
        s.skill_id + ': ' + (s.problem_solved || s.algorithm_summary || '').slice(0, 140)
      ).join('\n');
    }
    return top.map(s => {
      const p = [s.skill_id, s.title];
      if (s.problem_solved) p.push('解决: ' + s.problem_solved.slice(0, 120));
      if (s.biz_trigger) p.push('触发: ' + s.biz_trigger.slice(0, 100));
      if (s.roi_figure) p.push('ROI: ' + s.roi_figure);
      return p.join(' | ');
    }).join('\n');
  }

  function renderSkillCards(text) {
    const DATA = window.PLAYBOOK_DATA || {};
    const map = {};
    (DATA.skills || []).forEach(s => { map[s.skill_id] = s; });

    const found = [];
    const seen = {};
    [/\[\[?(Skill-[\w-]+)\]?\]/g, /\*\*(Skill-[\w-]+)\*\*/g].forEach(pat => {
      let m;
      while ((m = pat.exec(text)) !== null) {
        if (map[m[1]] && !seen[m[1]]) {
          seen[m[1]] = 1;
          found.push(map[m[1]]);
        }
      }
    });

    if (!found.length) return '';

    const esc = t => (t || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const cards = found.map(s =>
      '<a href="skills/' + s.skill_id + '.html" target="_blank" style="display:flex;align-items:flex-start;gap:10px;padding:10px 12px;background:var(--panel-2,#f8fafc);border:1px solid var(--line,#e2e8f0);border-radius:8px;text-decoration:none;color:inherit;margin-top:6px;transition:box-shadow .15s" onmouseover="this.style.boxShadow=\'0 2px 8px rgba(0,0,0,.08)\'" onmouseout="this.style.boxShadow=\'none\'">' +
      '<div style="flex-shrink:0;width:32px;height:32px;border-radius:6px;background:linear-gradient(135deg,#6366f1,#8b5cf6);display:flex;align-items:center;justify-content:center;color:#fff;font-size:11px;font-weight:700">S</div>' +
      '<div style="min-width:0">' +
      '<div style="font-size:12px;font-weight:600;color:#1e293b;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">' + esc((s.title || s.skill_id).slice(0, 60)) + '</div>' +
      '<div style="font-size:11.5px;color:#64748b;margin-top:2px;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical">' + esc((s.problem_solved || s.biz_trigger || '').slice(0, 90)) + '</div>' +
      (s.roi_figure ? '<span style="font-size:11px;color:#059669;font-weight:600;margin-top:4px;display:block">ROI: ' + esc(s.roi_figure) + '</span>' : '') +
      '</div></a>'
    ).join('');

    return '<div style="margin-top:10px;border-top:1px solid var(--line,#e2e8f0);padding-top:10px">' +
      '<div style="font-size:11.5px;color:#64748b;font-weight:600;margin-bottom:6px">知识库 相关技能</div>' +
      cards +
      '</div>';
  }

  const AKWS = {
    'agent-supply-sentinel': ['供应链', '库存', '断货', '补货', 'DOS', '海运'],
    'agent-pricing-advisor': ['定价', '价格', 'ACoS', '竞品价', '利润率'],
    'agent-pnl-analyzer': ['P&L', '利润', 'GMV', '毛利', '亏损'],
    'agent-ad-attribution': ['广告', 'ROAS', '归因', 'ACoS', '投放'],
    'agent-listing-doctor': ['Listing', '标题', '关键词', 'A+'],
    'agent-voc-decoder': ['评论', 'VOC', '用户反馈', '差评'],
    'agent-cs-triage': ['客服', '工单', '退款', '投诉', 'A-to-Z'],
    'agent-account-guardian': ['封号', '账号', '违规', '风险'],
    'agent-brand-guardian': ['合规', '文案', '广告法', '违禁'],
    'agent-product-radar': ['选品', '蓝海', '竞争', '市场机会'],
    'agent-tiktok-content': ['TikTok', '短视频', '内容', '脚本'],
    'agent-competitor-radar': ['竞品', '竞争对手', 'ASIN', 'BSR']
  };

  const ANAMES = {
    'agent-supply-sentinel': '供应链哨兵',
    'agent-pricing-advisor': '动态定价顾问',
    'agent-pnl-analyzer': 'P&L透视镜',
    'agent-ad-attribution': '广告归因侦探',
    'agent-listing-doctor': 'Listing医生',
    'agent-voc-decoder': '用户之声解码器',
    'agent-cs-triage': '客服分诊台',
    'agent-account-guardian': '账号风险卫士',
    'agent-brand-guardian': '品牌合规卫士',
    'agent-product-radar': '选品雷达',
    'agent-tiktok-content': 'TikTok内容官',
    'agent-competitor-radar': '竞品雷达站'
  };

  function detectAgents(text) {
    const t = text.toLowerCase();
    return Object.keys(AKWS).filter(id => AKWS[id].some(k => t.indexOf(k.toLowerCase()) >= 0)).slice(0, 3);
  }

  function renderAgentBtns(ids) {
    if (!ids.length) return '';
    const btns = ids.map(id =>
      '<a href="agents.html" target="_blank" style="display:inline-flex;align-items:center;gap:5px;padding:6px 12px;background:var(--accent-light,#eff6ff);border:1px solid var(--accent,#3b82f6);border-radius:20px;font-size:12px;font-weight:600;color:var(--accent,#3b82f6);text-decoration:none;transition:all .15s;white-space:nowrap" onmouseover="this.style.background=\'var(--accent,#3b82f6)\';this.style.color=\'#fff\'" onmouseout="this.style.background=\'var(--accent-light,#eff6ff)\';this.style.color=\'var(--accent,#3b82f6)\'">◈ ' + (ANAMES[id] || id) + '</a>'
    ).join('');
    return '<div style="margin-top:10px;display:flex;flex-wrap:wrap;gap:8px;border-top:1px solid var(--line,#e2e8f0);padding-top:10px"><span style="font-size:11.5px;color:#64748b;font-weight:600;align-self:center;margin-right:4px"> 直接调用：</span>' + btns + '</div>';
  }

  function addMsg(text, role, extras) {
    extras = extras || {};
    if (welcome) welcome.style.display = 'none';
    const row = document.createElement('div');
    row.className = 'cmsg cmsg-' + role;
    
    const av = document.createElement('div');
    av.className = 'cmsg-avatar';
    av.textContent = role === 'bot' ? '\u2726' : 'U';
    
    const body = document.createElement('div');
    body.className = 'cmsg-body';
    
    const nm = document.createElement('div');
    nm.className = 'cmsg-name';
    nm.textContent = role === 'bot' ? 'AI 助手' : '你';
    body.appendChild(nm);
    
    if (extras.webBadge) {
      const b = document.createElement('div');
      b.className = 'cmsg-web-badge';
      b.innerHTML = '联网搜索';
      body.appendChild(b);
    }
    
    if (extras.eventBadge) {
      const b = document.createElement('div');
      b.className = 'cmsg-event-badge';
      b.innerHTML = extras.eventBadge;
      body.appendChild(b);
    } else if (extras.ragBadge) {
      const b = document.createElement('div');
      b.className = 'cmsg-web-badge';
      b.style.cssText = 'background:#f0fdf4;color:#166534;border-color:#bbf7d0';
      b.innerHTML = '知识库检索 · ' + extras.ragBadge + ' 条相关技能';
      body.appendChild(b);
    }
    
    const bubble = document.createElement('div');
    bubble.className = 'cmsg-bubble';
    
    if (role === 'bot') {
      bubble.innerHTML = md(text);
      const agIds = detectAgents(text);
      const sc = renderSkillCards(text);
      const ab = renderAgentBtns(agIds);
      if (sc || ab) {
        const x = document.createElement('div');
        x.innerHTML = (sc || '') + (ab || '');
        bubble.appendChild(x);
      }
    } else {
      bubble.textContent = text;
    }
    
    body.appendChild(bubble);
    row.appendChild(av);
    row.appendChild(body);
    msgsEl.appendChild(row);
    msgsEl.scrollTop = msgsEl.scrollHeight;
    
    return { row, bubble };
  }

  function addTyping() {
    if (welcome) welcome.style.display = 'none';
    const row = document.createElement('div');
    row.className = 'cmsg cmsg-bot cmsg-typing';
    
    const av = document.createElement('div');
    av.className = 'cmsg-avatar';
    av.textContent = '\u2726';
    
    const body = document.createElement('div');
    body.className = 'cmsg-body';
    
    const nm = document.createElement('div');
    nm.className = 'cmsg-name';
    nm.textContent = 'AI 助手';
    
    const bubble = document.createElement('div');
    bubble.className = 'cmsg-bubble';
    
    body.appendChild(nm);
    body.appendChild(bubble);
    row.appendChild(av);
    row.appendChild(body);
    msgsEl.appendChild(row);
    msgsEl.scrollTop = msgsEl.scrollHeight;
    
    return row;
  }

  function matchRiskEvent(query) {
    if (!window.RISK_EVENTS || !window.RISK_EVENTS.events) return null;
    const lowerQuery = query.toLowerCase();
    let bestEvent = null;
    let maxScore = 0;
    
    for (const event of window.RISK_EVENTS.events) {
      if (!event.symptom_keywords) continue;
      let score = 0;
      for (const kw of event.symptom_keywords) {
        if (lowerQuery.includes(kw.toLowerCase())) {
          score++;
        }
      }
      if (score > maxScore) {
        maxScore = score;
        bestEvent = event;
      }
    }
    
    return maxScore > 0 ? bestEvent : null;
  }

  function buildEventSkillChain(event) {
    let result = '';
    const phases = event.phases || {};
    
    if (phases.diagnose && phases.diagnose.length > 0) {
      result += '【诊断层】\n';
      phases.diagnose.forEach((s, i) => {
        result += `  ${i+1}. ${s.skill_id}: ${s.role || ''}\n`;
      });
    }
    
    if (phases.treat && phases.treat.length > 0) {
      result += '【处置层】\n';
      phases.treat.forEach((s, i) => {
        const cond = s.condition ? `（条件: ${s.condition}时触发）` : '';
        result += `  ${i+1}. ${s.skill_id}: ${s.role || ''}${cond}\n`;
      });
    }
    
    if (phases.prevent && phases.prevent.length > 0) {
      result += '【预防层】\n';
      phases.prevent.forEach((s, i) => {
        result += `  ${i+1}. ${s.skill_id}: ${s.role || ''}\n`;
      });
    }
    
    return result;
  }

  async function doSend() {
    const text = textarea.value.trim();
    if (!text || sendBtn.disabled) return;
    textarea.value = '';
    textarea.style.height = 'auto';
    sendBtn.disabled = true;
    
    addMsg(text, 'user');
    history.push({ role: 'user', content: text });
    
    const typing = addTyping();
    
    let ctxMsg = '';
    const matchedEvent = matchRiskEvent(text);
    let matchedEventText = null;
    
    let sys = '你是 paper2skills 知识库的专业 AI 问答助手，专注于母婴跨境电商 AI 决策。\n知识库现有 {skill_count} 个从顶会论文萃取的可落地业务技能。\n回答规范：优先引用知识库中的具体 Skill，格式：[[Skill-具体名称]]；给出可操作具体建议。\n当前时间：' + new Date().toLocaleDateString('zh-CN', {year:'numeric',month:'long',day:'numeric'});
    
    if (matchedEvent) {
      sys += `\n\n当前诊断场景：${matchedEvent.event_name}\n严重程度：${matchedEvent.severity}`;
      const eventChain = buildEventSkillChain(matchedEvent);
      ctxMsg = '\n\n【场景推荐 Skill 链】\n' + eventChain;
      matchedEventText = `${matchedEvent.icon} 识别到场景：${matchedEvent.event_name}`;
    } else {
      const ragSkills = searchSkills(text, 10);
      const ragCtx = buildRAGContext(text);
      const ragCount = ragSkills.length;
      ctxMsg = ragCount > 0 ? '\n\n【知识库相关技能（检索到' + ragCount + '条）】\n' + ragCtx : '\n\n【知识库摘要（前60条）】\n' + ragCtx;
      const _idxSkills = await _retrieveSkills(text, 5);
      if (_idxSkills.length) {
        ctxMsg += '\n\n【知识库检索结果 — 请优先引用这些 Skill ID 回答】\n' +
          _idxSkills.map(function(s) {
            return '[' + s.id + '] ' + s.title + ': ' + s.summary.slice(0, 120);
          }).join('\n');
      }
    }
    
    const messages = [
      { role: 'system', content: sys + ctxMsg },
      ...history.slice(-8)
    ];
    
    try {
      const body = {
        model: 'deepseek-chat',
        messages,
        max_tokens: 1500,
        temperature: 0.55,
        stream: false
      };
      
      if (webSearchOn) {
        body.tools = [{ 
          type: 'function', 
          function: { 
            name: 'web_search', 
            description: 'Search the web', 
            parameters: { type: 'object', properties: { query: { type: 'string' } }, required: ['query'] } 
          } 
        }];
        body.tool_choice = 'auto';
      }
      
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      
      const data = await res.json();
      const choice = data && data.choices && data.choices[0];
      let answer = (choice && choice.message && choice.message.content || '').trim();
      
      if (!answer && choice && choice.finish_reason === 'tool_calls') {
        answer = '（联网搜索触发中…）\n\n' + ((choice.message.tool_calls[0] && choice.message.tool_calls[0].function.arguments) || '');
      }
      answer = answer || '抱歉，暂时无法获取回答，请稍后重试。';
      
      typing.remove();
      
      let ragCountToPass = null;
      let _lastRagSkills = [];
      if (!matchedEvent) {
          const ragSkills = searchSkills(text, 10);
          ragCountToPass = ragSkills.length > 0 ? ragSkills.length : null;
          _lastRagSkills = ragSkills;
      }
      
      const _msgResult = addMsg(answer, 'bot', { webBadge: webSearchOn, eventBadge: matchedEventText, ragBadge: ragCountToPass });
      
      if (_lastRagSkills.length) {
        _retrieveSkills(text, 5).then(function(idxSkills) {
          if (!idxSkills.length) return;
          const citDiv = document.createElement('div');
          citDiv.style.cssText = 'font-size:11px;color:#94a3b8;margin-top:6px;padding-top:6px;border-top:1px solid #f1f5f9';
          citDiv.innerHTML = '\u53c2\u8003 Skill: ' + idxSkills.map(function(s) {
            return '<a href="/skills/' + s.id + '.html" target="_blank" style="color:#6366f1;text-decoration:none">' + s.title + '</a>';
          }).join(' \u00b7 ');
          if (_msgResult && _msgResult.bubble) _msgResult.bubble.appendChild(citDiv);
        });
      }
      
      history.push({ role: 'assistant', content: answer });
      _saveH();
      
    } catch (e) {
      typing.remove();
      addMsg('网络请求失败，请检查连接后重试。', 'bot');
    } finally {
      sendBtn.disabled = false;
      textarea.focus();
    }
  }
})();
"""


def build_search_js() -> str:
    return r"""
(function(){
  const input = document.getElementById('global-search');
  const box   = document.getElementById('search-results');
  if (!input || !box || !window.PLAYBOOK_DATA) return;
  const skills = window.PLAYBOOK_DATA.skills || [];

  function applyFilters(list) {
    const diff  = (document.getElementById('filter-diff')  || {}).value || '';
    const roi   = (document.getElementById('filter-roi')   || {}).value || '';
    const dom   = (document.getElementById('filter-domain') || {}).value || '';
    return list.filter(s => {
      if (dom  && s.domain_dir !== dom) return false;
      if (diff && s.difficulty !== diff) return false;
      if (roi) {
        const stars = (s.difficulty || '').split('⭐').length - 1;
        if (roi === 'easy'   && stars > 2) return false;
        if (roi === 'medium' && (stars < 3 || stars > 3)) return false;
        if (roi === 'hard'   && stars < 4) return false;
      }
      return true;
    });
  }

  function doSearch() {
    const q = input.value.trim().toLowerCase();
    if (q.length < 2) { box.classList.add('hidden'); box.innerHTML = ''; return; }
    let hits = skills.filter(s =>
      [s.skill_id, s.title, s.domain_dir,
       (s.tags||[]).join(' '), (s.topics||[]).join(' '),
       s.algorithm_summary, s.problem_solved, s.roi_figure
      ].join(' ').toLowerCase().includes(q)
    );
    hits = applyFilters(hits).slice(0, 24);
    box.innerHTML = hits.map(s =>
      `<a class="result" href="${rootPrefix()}skills/${s.skill_id}.html">` +
      `<strong>${esc(s.title)}</strong>` +
      `<br><span>${esc(s.domain_dir)}` +
      `${s.roi_figure ? ' · ' + esc(s.roi_figure) : ''}` +
      `${s.difficulty ? ' · ' + esc(s.difficulty) : ''}</span></a>`
    ).join('') || '<p class="muted" style="padding:12px">无结果</p>';
    box.classList.remove('hidden');
  }

  input.addEventListener('input', doSearch);
  ['filter-diff','filter-roi','filter-domain'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('change', doSearch);
  });
  document.addEventListener('click', e => {
    if (e.target !== input && !box.contains(e.target)) box.classList.add('hidden');
  });
  function rootPrefix() {
    const p = location.pathname;
    return (p.includes('/skills/') || p.includes('/domains/') || p.includes('/topics/') ||
            p.includes('/workflows/') || p.includes('/playbooks/') || p.includes('/graph/')) ? '../' : '';
  }
  function esc(s) {
    return String(s||'').replace(/[&<>"']/g, c =>
      ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }
})();
""".strip()


# ---------------------------------------------------------------------------
# render_pages orchestrator
# ---------------------------------------------------------------------------

