with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'r', encoding='utf-8') as f:
    content = f.read()

# I see it generated the exact same thing again. Why? Oh, because I did git checkout previously.
# So I need to redo the changes.
import re

# 1. Add script tag to render_chat_page
html_replacement = r"""
  <script src="{nav}assets/playbook-data.js"></script>
  <script src="{nav}assets/risk-events.js"></script>
  <script src="{nav}assets/chat-page.js"></script>
"""
content = re.sub(r'  <script src="\{nav\}assets/playbook-data\.js"></script>\n  <script src="\{nav\}assets/chat-page\.js"></script>', html_replacement, content)

# 2. Update chat suggestions buttons
sug_replacement = r"""          <div class="chat-suggestions">
            <button class="chat-sug-btn">如何提升广告 ROI？</button>
            <button class="chat-sug-btn">大促备货如何预测需求？</button>
            <button class="chat-sug-btn">供应链 AI 有哪些关键技能？</button>
            <button class="chat-sug-btn">KOL 投放效果怎么归因？</button>
            <button class="chat-sug-btn">🔴 账号 ODR 异常，担心被封号</button>
            <button class="chat-sug-btn">📉 ASIN 流量突然下降 30%</button>
            <button class="chat-sug-btn">⚖️ 产品被平台合规警告</button>"""
content = re.sub(r'          <div class="chat-suggestions">.*?<button class="chat-sug-btn">用户流失预警方法有哪些？</button>', sug_replacement, content, flags=re.DOTALL)

# 3. Add styles for eventBadge
style_replacement = r"""    .cmsg-web-badge {
      display: inline-flex; align-items: center; gap: 4px;
      font-size: 11px; color: var(--muted); margin-bottom: 6px;
      padding: 2px 8px; background: var(--panel-2); border-radius: var(--r-full);
      border: 1px solid var(--line);
    }
    .cmsg-event-badge {
      display: inline-flex; align-items: center; gap: 4px;
      font-size: 11px; color: #991b1b; margin-bottom: 6px;
      padding: 2px 8px; background: #fef2f2; border-radius: var(--r-full);
      border: 1px solid #fecaca;
    }"""
content = re.sub(r'    \.cmsg-web-badge \{.*?border: 1px solid var\(--line\);\n    \}', style_replacement, content, flags=re.DOTALL)
# Fix brace escaping for f-string
content = content.replace("    .cmsg-web-badge {", "    .cmsg-web-badge {{")
content = content.replace("    .cmsg-event-badge {", "    .cmsg-event-badge {{")
content = content.replace("      border: 1px solid #fecaca;\n    }", "      border: 1px solid #fecaca;\n    }}")
content = content.replace("      border: 1px solid var(--line);\n    }", "      border: 1px solid var(--line);\n    }}")


start_js_idx = content.find('def build_chat_page_js() -> str:')
end_js_idx = content.find('def build_search_js() -> str:', start_js_idx)

js_block = content[start_js_idx:end_js_idx]

# Modify addMsg
js_block = js_block.replace(
    "if(extras.ragBadge){const b=document.createElement('div');b.className='cmsg-web-badge';b.style.cssText='background:#f0fdf4;color:#166534;border-color:#bbf7d0';b.innerHTML='知识库检索 · '+extras.ragBadge+' 条相关技能';body.appendChild(b);}",
    "if(extras.eventBadge){const b=document.createElement('div');b.className='cmsg-event-badge';b.innerHTML=extras.eventBadge;body.appendChild(b);}else if(extras.ragBadge){const b=document.createElement('div');b.className='cmsg-web-badge';b.style.cssText='background:#f0fdf4;color:#166534;border-color:#bbf7d0';b.innerHTML='知识库检索 · '+extras.ragBadge+' 条相关技能';body.appendChild(b);}"
)


new_functions = r"""
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
      result += '【诊断层】\\n';
      phases.diagnose.forEach((s, i) => {
        result += `  ${i+1}. ${s.skill_id}: ${s.role || ''}\\n`;
      });
    }
    
    if (phases.treat && phases.treat.length > 0) {
      result += '【处置层】\\n';
      phases.treat.forEach((s, i) => {
        const cond = s.condition ? `（条件: ${s.condition}时触发）` : '';
        result += `  ${i+1}. ${s.skill_id}: ${s.role || ''}${cond}\\n`;
      });
    }
    
    if (phases.prevent && phases.prevent.length > 0) {
      result += '【预防层】\\n';
      phases.prevent.forEach((s, i) => {
        result += `  ${i+1}. ${s.skill_id}: ${s.role || ''}\\n`;
      });
    }
    return result;
  }

  async function doSend(){"""

js_block = js_block.replace("  async function doSend(){", new_functions)

do_send_old = r"""const ragSkills=searchSkills(text,10),ragCtx=buildRAGContext(text),ragCount=ragSkills.length;const sys='你是 paper2skills 知识库的专业 AI 问答助手，专注于母婴跨境电商 AI 决策。\n知识库现有 {skill_count} 个从顶会论文萃取的可落地业务技能。\n回答规范：优先引用知识库中的具体 Skill，格式：[[Skill-具体名称]]；给出可操作具体建议。\n当前时间：'+new Date().toLocaleDateString('zh-CN',{year:'numeric',month:'long',day:'numeric'});const ctxMsg=ragCount>0?'\n\n【知识库相关技能（检索到'+ragCount+'条）】\n'+ragCtx:'\n\n【知识库摘要（前60条）】\n'+ragCtx;const messages=[{role:'system',content:sys+ctxMsg},...history.slice(-8)];"""

do_send_new = r"""let ctxMsg = '';
  const matchedEvent = matchRiskEvent(text);
  let matchedEventText = null;
  let sys='你是 paper2skills 知识库的专业 AI 问答助手，专注于母婴跨境电商 AI 决策。\\n知识库现有 {skill_count} 个从顶会论文萃取的可落地业务技能。\\n回答规范：优先引用知识库中的具体 Skill，格式：[[Skill-具体名称]]；给出可操作具体建议。\\n当前时间：'+new Date().toLocaleDateString('zh-CN',{year:'numeric',month:'long',day:'numeric'});
  if (matchedEvent) {
    sys += `\\n\\n当前诊断场景：${matchedEvent.event_name}\\n严重程度：${matchedEvent.severity}`;
    const eventChain = buildEventSkillChain(matchedEvent);
    ctxMsg = '\\n\\n【场景推荐 Skill 链】\\n' + eventChain;
    matchedEventText = `${matchedEvent.icon} 识别到场景：${matchedEvent.event_name}`;
  } else {
    const ragSkills=searchSkills(text,10),ragCtx=buildRAGContext(text),ragCount=ragSkills.length;
    ctxMsg=ragCount>0?'\\n\\n【知识库相关技能（检索到'+ragCount+'条）】\\n'+ragCtx:'\\n\\n【知识库摘要（前60条）】\\n'+ragCtx;
  }
  const messages=[{role:'system',content:sys+ctxMsg},...history.slice(-8)];"""
js_block = js_block.replace(do_send_old, do_send_new)

js_block = js_block.replace("addMsg(answer,'bot',{webBadge:webSearchOn,ragBadge:ragCount>0?ragCount:null});", "addMsg(answer,'bot',{webBadge:webSearchOn, eventBadge: matchedEventText, ragBadge: (typeof ragCount !== 'undefined' && ragCount>0)?ragCount:null});")

content = content[:start_js_idx] + js_block + content[end_js_idx:]

with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'w', encoding='utf-8') as f:
    f.write(content)
