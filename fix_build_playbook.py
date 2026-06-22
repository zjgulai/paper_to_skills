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

# 4. Modify build_chat_page_js
# Using string replace instead of regex sub for the large function body to avoid escape sequence issues
start_marker = "  function md(text) {"
end_marker = "      textarea.focus();\n    }\n  }"

js_replacement = """  function md(text) {
    return text
      .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
      .replace(/\\*\\*(.+?)\\*\\*/gs,'<strong>$1</strong>')
      .replace(/\\*([^*\\n]+)\\*/g,'<em>$1</em>')
      .replace(/`([^`\\n]+)`/g,'<code>$1</code>')
      .replace(/^#{1,3}\\s+(.+)$/gm,'<strong style="font-size:15px">$1</strong>')
      .replace(/^[-•]\\s+(.+)$/gm,'<span style="display:block;padding-left:14px;margin:2px 0">• $1</span>')
      .replace(/^\\d+\\.\\s+(.+)$/gm,'<span style="display:block;padding-left:14px;margin:2px 0">$&</span>')
      .replace(/\\n\\n+/g,'<br><br>').replace(/\\n/g,'<br>');
  }

  function addMsg(text, role, webBadge, eventBadge) {
    if (welcome) welcome.style.display = 'none';
    const row = document.createElement('div');
    row.className = 'cmsg cmsg-' + role;
    const avatar = document.createElement('div');
    avatar.className = 'cmsg-avatar';
    avatar.textContent = role === 'bot' ? '\\u2726' : 'U';
    const body = document.createElement('div');
    body.className = 'cmsg-body';
    const name = document.createElement('div');
    name.className = 'cmsg-name';
    name.textContent = role === 'bot' ? 'AI 助手' : '你';
    body.appendChild(name);
    if (webBadge) {
      const badge = document.createElement('div');
      badge.className = 'cmsg-web-badge';
      badge.innerHTML = '联网搜索';
      body.appendChild(badge);
    }
    if (eventBadge) {
      const badge = document.createElement('div');
      badge.className = 'cmsg-event-badge';
      badge.innerHTML = eventBadge;
      body.appendChild(badge);
    }
    const bubble = document.createElement('div');
    bubble.className = 'cmsg-bubble';
    if (role === 'bot') { bubble.innerHTML = md(text); }
    else { bubble.textContent = text; }
    body.appendChild(bubble);
    row.appendChild(avatar);
    row.appendChild(body);
    msgsEl.appendChild(row);
    msgsEl.scrollTop = msgsEl.scrollHeight;
    return { row, bubble };
  }

  function addTyping() {
    if (welcome) welcome.style.display = 'none';
    const row = document.createElement('div');
    row.className = 'cmsg cmsg-bot cmsg-typing';
    const avatar = document.createElement('div');
    avatar.className = 'cmsg-avatar';
    avatar.textContent = '\\u2726';
    const body = document.createElement('div');
    body.className = 'cmsg-body';
    const name = document.createElement('div');
    name.className = 'cmsg-name';
    name.textContent = 'AI 助手';
    const bubble = document.createElement('div');
    bubble.className = 'cmsg-bubble';
    body.appendChild(name);
    body.appendChild(bubble);
    row.appendChild(avatar);
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

  function buildContext() {
    const DATA = window.PLAYBOOK_DATA || {};
    return (DATA.skills || []).slice(0, 80).map(s =>
      s.skill_id + ': ' + (s.problem_solved || s.algorithm_summary || '').slice(0, 140)
    ).join('\\n');
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
    ];

    try {
      const body = {
        model: 'deepseek-chat',
        messages,
        max_tokens: 1200,
        temperature: 0.6,
        stream: false
      };
      if (webSearchOn) {
        body.tools = [{ type: 'function', function: { name: 'web_search', description: 'Search the web for current information', parameters: { type: 'object', properties: { query: { type: 'string' } }, required: ['query'] } } }];
        body.tool_choice = 'auto';
      }
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
      });
      const data = await res.json();
      const choice = data?.choices?.[0];
      let answer = choice?.message?.content?.trim();
      if (!answer && choice?.finish_reason === 'tool_calls') {
        answer = '（已触发联网搜索，DeepSeek 正在整合结果…）\\n\\n' + (choice?.message?.tool_calls?.[0]?.function?.arguments || '');
      }
      answer = answer || '抱歉，暂时无法获取回答，请稍后重试。';
      typing.remove();
      addMsg(answer, 'bot', webSearchOn, matchedEventText);
      history.push({ role: 'assistant', content: answer });
    } catch (e) {
      typing.remove();
      addMsg('网络请求失败，请检查连接后重试。', 'bot');
    } finally {
      sendBtn.disabled = false;
      textarea.focus();
    }
  }"""

start_idx = content.find(start_marker)
end_idx = content.find(end_marker) + len(end_marker)

if start_idx != -1 and end_idx != -1:
    content = content[:start_idx] + js_replacement + content[end_idx:]

with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Replacement done.")
