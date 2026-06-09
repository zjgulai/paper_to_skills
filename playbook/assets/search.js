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