(function () {
  const msgsEl=document.getElementById('chat-messages'),welcome=document.getElementById('chat-welcome'),textarea=document.getElementById('chat-input'),sendBtn=document.getElementById('chat-send'),webToggle=document.getElementById('web-search-toggle'),webLabel=document.getElementById('web-search-label');
  let webSearchOn=false;
  const _HIST_KEY='p2s_chat_v1';
  function _loadH(){try{return JSON.parse(localStorage.getItem(_HIST_KEY)||'[]');}catch(e){return[];}}
  function _saveH(){try{localStorage.setItem(_HIST_KEY,JSON.stringify(history.slice(-20)));}catch(e){}}
  let history=_loadH();
  window.clearHistory=function(){history=[];try{localStorage.removeItem(_HIST_KEY);}catch(e){}if(msgsEl){msgsEl.innerHTML='';}if(welcome)welcome.style.display='';};
  webToggle.addEventListener('click',()=>{webSearchOn=!webSearchOn;webToggle.classList.toggle('on',webSearchOn);webLabel.textContent=webSearchOn?'已开启联网':'联网搜索';});
  textarea.addEventListener('input',()=>{textarea.style.height='auto';textarea.style.height=Math.min(textarea.scrollHeight,160)+'px';});
  textarea.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();doSend();}});
  sendBtn.addEventListener('click',doSend);
  if(history.length&&welcome){welcome.style.display='none';history.forEach(function(m){if(m.role==='user')addMsg(m.content,'user');else if(m.role==='assistant')addMsg(m.content,'bot');});}
  document.querySelectorAll('.chat-sug-btn').forEach(btn=>{btn.addEventListener('click',()=>{textarea.value=btn.textContent.trim();textarea.dispatchEvent(new Event('input'));doSend();});});
  function md(text){return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/\*\*(.+?)\*\*/gs,'<strong>$1</strong>').replace(/\*([^*\n]+)\*/g,'<em>$1</em>').replace(/`([^`\n]+)`/g,'<code>$1</code>').replace(/^#{1,3}\s+(.+)$/gm,'<strong style="font-size:15px">$1</strong>').replace(/^[-•]\s+(.+)$/gm,'<span style="display:block;padding-left:14px;margin:2px 0">• $1</span>').replace(/^\d+\.\s+(.+)$/gm,'<span style="display:block;padding-left:14px;margin:2px 0">$&</span>').replace(/\n\n+/g,'<br><br>').replace(/\n/g,'<br>');}
  const _idx=[];let _built=false;
  function buildSkillIndex(){if(_built)return;const DATA=window.PLAYBOOK_DATA||{};(DATA.skills||[]).forEach(s=>{const t=[s.skill_id||'',s.title||'',s.problem_solved||'',s.algorithm_summary||'',s.biz_trigger||'',s.biz_outcome||'',(s.tags||[]).join(' '),(s.topics||[]).join(' ')].join(' ').toLowerCase();_idx.push({s,t});});_built=true;}
  let _skillIdx=null;async function _loadSkillIdx(){if(_skillIdx)return _skillIdx;try{_skillIdx=await fetch('/assets/skill-index.json').then(r=>r.json());}catch(e){_skillIdx=[];}return _skillIdx;}
  async function _retrieveSkills(query,topK){topK=topK||5;const idx=await _loadSkillIdx();if(!idx||!idx.length)return[];const tokens=query.toLowerCase().replace(/[^\u4e00-\u9fa5a-z0-9\s]/g,' ').split(/\s+/).filter(t=>t.length>1);if(!tokens.length)return[];const scored=idx.map(function(s){var sc=0;tokens.forEach(function(t){if((s.title||'').toLowerCase().includes(t))sc+=8;if((s.problem_solved||'').toLowerCase().includes(t))sc+=5;if((s.biz_trigger||'').toLowerCase().includes(t))sc+=4;if((s.biz_outcome||'').toLowerCase().includes(t))sc+=3;if((s.summary||'').toLowerCase().includes(t))sc+=2;if((s.keywords||[]).join(' ').toLowerCase().includes(t))sc+=1;});return{s,score:sc};}).filter(x=>x.score>0);scored.sort((a,b)=>b.score-a.score);return scored.slice(0,topK).map(x=>x.s);}
  function searchSkills(query,k){k=k||8;buildSkillIndex();const words=query.toLowerCase().split(/\s+/).filter(w=>w.length>1);if(!words.length)return[];return _idx.map(item=>{let sc=0;words.forEach(w=>{const tf=item.t.split(w).length-1;if(tf>0)sc+=tf*(w.length>3?2:1);});return{skill:item.s,sc};}).filter(x=>x.sc>0).sort((a,b)=>b.sc-a.sc).slice(0,k).map(x=>x.skill);}
  function buildRAGContext(query){const top=searchSkills(query,10);if(!top.length){return(window.PLAYBOOK_DATA&&window.PLAYBOOK_DATA.skills||[]).slice(0,60).map(s=>s.skill_id+': '+(s.problem_solved||s.algorithm_summary||'').slice(0,140)).join('\n');}return top.map(s=>{const p=[s.skill_id,s.title];if(s.problem_solved)p.push('解决: '+s.problem_solved.slice(0,120));if(s.biz_trigger)p.push('触发: '+s.biz_trigger.slice(0,100));if(s.roi_figure)p.push('ROI: '+s.roi_figure);return p.join(' | ');}).join('\n');}
  function renderSkillCards(text){const DATA=window.PLAYBOOK_DATA||{};const map={};(DATA.skills||[]).forEach(s=>{map[s.skill_id]=s;});const found=[],seen={};[/\[\[?(Skill-[\w-]+)\]?\]/g,/\*\*(Skill-[\w-]+)\*\*/g].forEach(pat=>{let m;while((m=pat.exec(text))!==null){if(map[m[1]]&&!seen[m[1]]){seen[m[1]]=1;found.push(map[m[1]]);}}});if(!found.length)return'';const esc=t=>(t||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');const cards=found.map(s=>'<a href="skills/'+s.skill_id+'.html" target="_blank" style="display:flex;align-items:flex-start;gap:10px;padding:10px 12px;background:var(--panel-2,#f8fafc);border:1px solid var(--line,#e2e8f0);border-radius:8px;text-decoration:none;color:inherit;margin-top:6px;transition:box-shadow .15s" onmouseover="this.style.boxShadow=\'0 2px 8px rgba(0,0,0,.08)\'" onmouseout="this.style.boxShadow=\'none\'">'+'<div style="flex-shrink:0;width:32px;height:32px;border-radius:6px;background:linear-gradient(135deg,#6366f1,#8b5cf6);display:flex;align-items:center;justify-content:center;color:#fff;font-size:11px;font-weight:700">S</div>'+'<div style="min-width:0"><div style="font-size:12px;font-weight:600;color:#1e293b;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">'+esc((s.title||s.skill_id).slice(0,60))+'</div>'+'<div style="font-size:11.5px;color:#64748b;margin-top:2px;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical">'+esc((s.problem_solved||s.biz_trigger||'').slice(0,90))+'</div>'+(s.roi_figure?'<span style="font-size:11px;color:#059669;font-weight:600;margin-top:4px;display:block">ROI: '+esc(s.roi_figure)+'</span>':'')+'</div></a>').join('');return'<div style="margin-top:10px;border-top:1px solid var(--line,#e2e8f0);padding-top:10px"><div style="font-size:11.5px;color:#64748b;font-weight:600;margin-bottom:6px">知识库 相关技能</div>'+cards+'</div>';}
  const AKWS={'agent-supply-sentinel':['供应链','库存','断货','补货','DOS','海运'],'agent-pricing-advisor':['定价','价格','ACoS','竞品价','利润率'],'agent-pnl-analyzer':['P&L','利润','GMV','毛利','亏损'],'agent-ad-attribution':['广告','ROAS','归因','ACoS','投放'],'agent-listing-doctor':['Listing','标题','关键词','A+'],'agent-voc-decoder':['评论','VOC','用户反馈','差评'],'agent-cs-triage':['客服','工单','退款','投诉','A-to-Z'],'agent-account-guardian':['封号','账号','违规','风险'],'agent-brand-guardian':['合规','文案','广告法','违禁'],'agent-product-radar':['选品','蓝海','竞争','市场机会'],'agent-tiktok-content':['TikTok','短视频','内容','脚本'],'agent-competitor-radar':['竞品','竞争对手','ASIN','BSR']};
  const ANAMES={'agent-supply-sentinel':'供应链哨兵','agent-pricing-advisor':'动态定价顾问','agent-pnl-analyzer':'P&L透视镜','agent-ad-attribution':'广告归因侦探','agent-listing-doctor':'Listing医生','agent-voc-decoder':'用户之声解码器','agent-cs-triage':'客服分诊台','agent-account-guardian':'账号风险卫士','agent-brand-guardian':'品牌合规卫士','agent-product-radar':'选品雷达','agent-tiktok-content':'TikTok内容官','agent-competitor-radar':'竞品雷达站'};
  function detectAgents(text){const t=text.toLowerCase();return Object.keys(AKWS).filter(id=>AKWS[id].some(k=>t.indexOf(k.toLowerCase())>=0)).slice(0,3);}
  function renderAgentBtns(ids){if(!ids.length)return'';const btns=ids.map(id=>'<a href="agents.html" target="_blank" style="display:inline-flex;align-items:center;gap:5px;padding:6px 12px;background:var(--accent-light,#eff6ff);border:1px solid var(--accent,#3b82f6);border-radius:20px;font-size:12px;font-weight:600;color:var(--accent,#3b82f6);text-decoration:none;transition:all .15s;white-space:nowrap" onmouseover="this.style.background=\'var(--accent,#3b82f6)\';this.style.color=\'#fff\'" onmouseout="this.style.background=\'var(--accent-light,#eff6ff)\';this.style.color=\'var(--accent,#3b82f6)\'">◈ '+(ANAMES[id]||id)+'</a>').join('');return'<div style="margin-top:10px;display:flex;flex-wrap:wrap;gap:8px;border-top:1px solid var(--line,#e2e8f0);padding-top:10px"><span style="font-size:11.5px;color:#64748b;font-weight:600;align-self:center;margin-right:4px"> 直接调用：</span>'+btns+'</div>';}
  function addMsg(text,role,extras){extras=extras||{};if(welcome)welcome.style.display='none';const row=document.createElement('div');row.className='cmsg cmsg-'+role;const av=document.createElement('div');av.className='cmsg-avatar';av.textContent=role==='bot'?'\u2726':'U';const body=document.createElement('div');body.className='cmsg-body';const nm=document.createElement('div');nm.className='cmsg-name';nm.textContent=role==='bot'?'AI 助手':'你';body.appendChild(nm);if(extras.webBadge){const b=document.createElement('div');b.className='cmsg-web-badge';b.innerHTML='联网搜索';body.appendChild(b);}if(extras.eventBadge){const b=document.createElement('div');b.className='cmsg-event-badge';b.innerHTML=extras.eventBadge;body.appendChild(b);}else if(extras.ragBadge){const b=document.createElement('div');b.className='cmsg-web-badge';b.style.cssText='background:#f0fdf4;color:#166534;border-color:#bbf7d0';b.innerHTML='知识库检索 · '+extras.ragBadge+' 条相关技能';body.appendChild(b);}const bubble=document.createElement('div');bubble.className='cmsg-bubble';if(role==='bot'){bubble.innerHTML=md(text);const agIds=detectAgents(text),sc=renderSkillCards(text),ab=renderAgentBtns(agIds);if(sc||ab){const x=document.createElement('div');x.innerHTML=(sc||'')+(ab||'');bubble.appendChild(x);}}else{bubble.textContent=text;}body.appendChild(bubble);row.appendChild(av);row.appendChild(body);msgsEl.appendChild(row);msgsEl.scrollTop=msgsEl.scrollHeight;return{row,bubble};}
  function addTyping(){if(welcome)welcome.style.display='none';const row=document.createElement('div');row.className='cmsg cmsg-bot cmsg-typing';const av=document.createElement('div');av.className='cmsg-avatar';av.textContent='\u2726';const body=document.createElement('div');body.className='cmsg-body';const nm=document.createElement('div');nm.className='cmsg-name';nm.textContent='AI 助手';const bubble=document.createElement('div');bubble.className='cmsg-bubble';body.appendChild(nm);body.appendChild(bubble);row.appendChild(av);row.appendChild(body);msgsEl.appendChild(row);msgsEl.scrollTop=msgsEl.scrollHeight;return row;}

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

  async function doSend(){const text=textarea.value.trim();if(!text||sendBtn.disabled)return;textarea.value='';textarea.style.height='auto';sendBtn.disabled=true;addMsg(text,'user');history.push({role:'user',content:text});const typing=addTyping();
  
  let ctxMsg = '';
  const matchedEvent = matchRiskEvent(text);
  let matchedEventText = null;
  
  const roleSelect = document.getElementById('role-select');
  const roleVal = roleSelect ? roleSelect.value : 'ops';
  const rolePrompts = {
    ops: '当前用户是电商运营，关注具体操作步骤、SOP 执行、数据指标改善，回答要简洁可执行。',
    analyst: '当前用户是数据分析师，关注算法原理、统计方法、代码实现，回答要有技术深度，可附公式。',
    ceo: '当前用户是 CEO，关注战略决策、ROI 全局、竞争壁垒，回答要高度概括、突出商业价值。',
  };
  const roleCtx = rolePrompts[roleVal] || rolePrompts.ops;
  
  let sys=`你是 paper2skills 知识库的专业 AI 问答助手，专注于母婴跨境电商 AI 决策。\n知识库现有 ${window.PLAYBOOK_DATA && window.PLAYBOOK_DATA.skills ? window.PLAYBOOK_DATA.skills.length : 800}+ 个从顶会论文萃取的可落地业务技能。\n回答规范：优先引用知识库中的具体 Skill，格式：[[Skill-具体名称]]；给出可操作具体建议。\n${roleCtx}\n当前时间：${new Date().toLocaleDateString('zh-CN',{year:'numeric',month:'long',day:'numeric'})}`;
  
  if (matchedEvent) {
    sys += `\n\n当前诊断场景：${matchedEvent.event_name}\n严重程度：${matchedEvent.severity}`;
    const eventChain = buildEventSkillChain(matchedEvent);
    ctxMsg = '\n\n【场景推荐 Skill 链】\n' + eventChain;
    matchedEventText = `${matchedEvent.icon} 识别到场景：${matchedEvent.event_name}`;
  } else {
    const ragSkills=searchSkills(text,10),ragCtx=buildRAGContext(text),ragCount=ragSkills.length;
    ctxMsg=ragCount>0?'\n\n【知识库相关技能（检索到'+ragCount+'条）】\n'+ragCtx:'\n\n【知识库摘要（前60条）】\n'+ragCtx;
    const _idxSkills=await _retrieveSkills(text,5);if(_idxSkills.length){ctxMsg+='\n\n【知识库检索结果 — 请优先引用这些 Skill ID 回答】\n'+_idxSkills.map(function(s){return'['+s.id+'] '+s.title+': '+s.summary.slice(0,120);}).join('\n');}
  }
  
  const messages=[{role:'system',content:sys+ctxMsg},...history.slice(-8)];try{const body={model:'deepseek-chat',messages,max_tokens:1500,temperature:0.55,stream:false};if(webSearchOn){body.tools=[{type:'function',function:{name:'web_search',description:'Search the web',parameters:{type:'object',properties:{query:{type:'string'}},required:['query']}}}];body.tool_choice='auto';}const res=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const data=await res.json();const choice=data&&data.choices&&data.choices[0];let answer=(choice&&choice.message&&choice.message.content||'').trim();if(!answer&&choice&&choice.finish_reason==='tool_calls')answer='（联网搜索触发中…）\n\n'+((choice.message.tool_calls[0]&&choice.message.tool_calls[0].function.arguments)||'');answer=answer||'抱歉，暂时无法获取回答，请稍后重试。';typing.remove();
  let ragCountToPass = null;
  if (!matchedEvent) {
      const ragSkills = searchSkills(text, 10);
      ragCountToPass = ragSkills.length > 0 ? ragSkills.length : null;
  }
  const _msgResult=addMsg(answer,'bot',{webBadge:webSearchOn, eventBadge: matchedEventText, ragBadge: ragCountToPass});
  if (!matchedEvent){_retrieveSkills(text,5).then(function(idxSkills){if(!idxSkills.length)return;const citDiv=document.createElement('div');citDiv.style.cssText='font-size:11px;color:#94a3b8;margin-top:6px;padding-top:6px;border-top:1px solid #f1f5f9';citDiv.innerHTML='\u53c2\u8003 Skill: '+idxSkills.map(function(s){return'<a href="/skills/'+s.id+'.html" target="_blank" style="color:#6366f1;text-decoration:none">'+s.title+'</a>';}).join(' \u00b7 ');if(_msgResult&&_msgResult.bubble)_msgResult.bubble.appendChild(citDiv);});}
  history.push({role:'assistant',content:answer});_saveH();}catch(e){typing.remove();addMsg('网络请求失败，请检查连接后重试。','bot');}finally{sendBtn.disabled=false;textarea.focus();}}
})();
