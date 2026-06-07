(function(){
  const input=document.getElementById('global-search');
  const box=document.getElementById('search-results');
  if(!input||!box||!window.PLAYBOOK_DATA)return;
  const skills=window.PLAYBOOK_DATA.skills||[];
  input.addEventListener('input',()=>{
    const q=input.value.trim().toLowerCase();
    if(q.length<2){box.classList.add('hidden');box.innerHTML='';return;}
    const hits=skills.filter(s=>[s.skill_id,s.title,s.domain_dir,(s.tags||[]).join(' '),(s.topics||[]).join(' '),s.algorithm_summary,s.problem_solved].join(' ').toLowerCase().includes(q)).slice(0,20);
    box.innerHTML=hits.map(s=>`<a class="result" href="${rootPrefix()}skills/${s.skill_id}.html"><strong>${escapeHtml(s.title)}</strong><br><span>${escapeHtml(s.domain_dir)} · ${escapeHtml(s.skill_id)}</span></a>`).join('')||'<p class="muted">无结果</p>';
    box.classList.remove('hidden');
  });
  document.addEventListener('click',e=>{if(e.target!==input&&!box.contains(e.target))box.classList.add('hidden')});
  function rootPrefix(){const path=location.pathname; if(path.includes('/skills/')||path.includes('/domains/')||path.includes('/topics/')||path.includes('/workflows/')||path.includes('/graph/'))return '../'; return '';}
  function escapeHtml(s){return String(s||'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
})();