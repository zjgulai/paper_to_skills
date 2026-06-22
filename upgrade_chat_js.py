import re

with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'r', encoding='utf-8') as f:
    content = f.read()

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

# 4. Modify build_chat_page_js
# Extract build_chat_page_js block
start_js_idx = content.find('def build_chat_page_js() -> str:')
end_js_idx = content.find('def build_search_js() -> str:', start_js_idx)
js_block = content[start_js_idx:end_js_idx]

# Perform JS updates inside js_block
js_block = js_block.replace("function addMsg(text, role, webBadge) {", "function addMsg(text, role, webBadge, eventBadge) {")
js_block = js_block.replace("if (webBadge) {", "if (eventBadge) {\n      const badge = document.createElement('div');\n      badge.className = 'cmsg-event-badge';\n      badge.innerHTML = eventBadge;\n      body.appendChild(badge);\n    }\n    if (webBadge) {")

route_functions = r"""
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

  function buildContext() {"""
js_block = js_block.replace("  function buildContext() {", route_functions)

dosend_replace_from = """    const typing = addTyping();
    const ctx = buildContext();
    const systemPrompt = `你是 paper2skills 知识库的专业 AI 问答助手，专注于母婴跨境电商 AI 决策技能。知识库收录了{skill_count}个从顶会论文（NeurIPS/KDD/ICML/WWW）萃取的可落地业务技能，涵盖供应链优化、广告归因、用户分析、KOL投放、合规决策、智能体工程等领域。请用清晰、结构化的中文回答，优先引用知识库中的具体Skill，给出可操作建议。当前时间：${new Date().toLocaleDateString('zh-CN', {year:'numeric',month:'long',day:'numeric'})}。`;

    const messages = [
      { role: 'system', content: systemPrompt + '\\n\\n知识库摘要（前80条Skill）：\\n' + ctx },
      ...history.slice(-6)
    ];"""

dosend_replace_to = """    const typing = addTyping();
    let ctx = '';
    let systemPrompt = `你是 paper2skills 知识库的专业 AI 问答助手，专注于母婴跨境电商 AI 决策技能。知识库收录了{skill_count}个从顶会论文（NeurIPS/KDD/ICML/WWW）萃取的可落地业务技能，涵盖供应链优化、广告归因、用户分析、KOL投放、合规决策、智能体工程等领域。请用清晰、结构化的中文回答，优先引用知识库中的具体Skill，给出可操作建议。当前时间：${new Date().toLocaleDateString('zh-CN', {year:'numeric',month:'long',day:'numeric'})}。`;
    let matchedEventText = null;

    const matchedEvent = matchRiskEvent(text);
    if (matchedEvent) {
      systemPrompt += `\\n\\n当前诊断场景：${matchedEvent.event_name}\\n严重程度：${matchedEvent.severity}`;
      ctx = buildEventSkillChain(matchedEvent);
      matchedEventText = `${matchedEvent.icon} 识别到场景：${matchedEvent.event_name}`;
    } else {
      ctx = buildContext();
    }

    const messages = [
      { role: 'system', content: systemPrompt + (matchedEvent ? '\\n\\n场景推荐 Skill 链：\\n' + ctx : '\\n\\n知识库摘要（前80条Skill）：\\n' + ctx) },
      ...history.slice(-6)
    ];"""
js_block = js_block.replace(dosend_replace_from, dosend_replace_to)

js_block = js_block.replace("addMsg(answer, 'bot', webSearchOn);", "addMsg(answer, 'bot', webSearchOn, matchedEventText);")

content = content[:start_js_idx] + js_block + content[end_js_idx:]

with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Modification done.")
