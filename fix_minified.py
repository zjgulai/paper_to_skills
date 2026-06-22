with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'r', encoding='utf-8') as f:
    content = f.read()

# It seems `build_chat_page_js` contains a minified version of the JS. Let's find it.
import re
start_js_idx = content.find('def build_chat_page_js() -> str:')
end_js_idx = content.find('def build_search_js() -> str:', start_js_idx)

js_block = content[start_js_idx:end_js_idx]

# In the minified version:
# function addMsg(text,role,extras){
js_block = js_block.replace(
    "if(extras.ragBadge){const b=document.createElement('div');b.className='cmsg-web-badge';b.style.cssText='background:#f0fdf4;color:#166534;border-color:#bbf7d0';b.innerHTML='知识库检索 · '+extras.ragBadge+' 条相关技能';body.appendChild(b);}",
    "if(extras.eventBadge){const b=document.createElement('div');b.className='cmsg-event-badge';b.innerHTML=extras.eventBadge;body.appendChild(b);}else if(extras.ragBadge){const b=document.createElement('div');b.className='cmsg-web-badge';b.style.cssText='background:#f0fdf4;color:#166534;border-color:#bbf7d0';b.innerHTML='知识库检索 · '+extras.ragBadge+' 条相关技能';body.appendChild(b);}"
)

# Insert matchRiskEvent and buildEventSkillChain
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

# Update doSend
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

