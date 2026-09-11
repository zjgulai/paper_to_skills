(function () {
  const svg = document.getElementById('ego-graph');
  if (!svg || typeof d3 === 'undefined') return;
  const centerId = svg.dataset.skill;
  if (!centerId) return;

  const W = +svg.getAttribute('width')  || 280;
  const H = +svg.getAttribute('height') || 220;

  function load(cb) {
    if (window._EGO_DATA) { cb(window._EGO_DATA); return; }
    const xhr = new XMLHttpRequest();
    xhr.open('GET', '../assets/graph-data.json');
    xhr.onload = () => {
      try { window._EGO_DATA = JSON.parse(xhr.responseText); cb(window._EGO_DATA); }
      catch (e) { cb(null); }
    };
    xhr.onerror = () => cb(null);
    xhr.send();
  }

  load(function (raw) {
    if (!raw) return;

    const edgeCfg = {
      prerequisite: '#3b82f6',
      combinable:   '#10b981',
      extension:    '#f59e0b',
    };

    const neighborIds = new Set([centerId]);
    const egoLinks = raw.links.filter(l => {
      if (l.source === centerId || l.target === centerId) {
        neighborIds.add(l.source);
        neighborIds.add(l.target);
        return true;
      }
      return false;
    });

    if (neighborIds.size <= 1) {
      d3.select(svg).append('text')
        .attr('x', W / 2).attr('y', H / 2)
        .attr('text-anchor', 'middle').attr('fill', '#9ca3af').attr('font-size', 12)
        .text('无关联 Skill');
      return;
    }

    const egoNodes = raw.nodes
      .filter(n => neighborIds.has(n.id))
      .map(n => ({ ...n }));

    const sel = d3.select(svg).attr('viewBox', `0 0 ${W} ${H}`);

    const sim = d3.forceSimulation(egoNodes)
      .force('link', d3.forceLink(egoLinks.map(l => ({ ...l }))).id(d => d.id).distance(65).strength(0.6))
      .force('charge', d3.forceManyBody().strength(-120))
      .force('center', d3.forceCenter(W / 2, H / 2))
      .force('collide', d3.forceCollide(18));

    const linkEl = sel.append('g').selectAll('line')
      .data(egoLinks)
      .join('line')
      .attr('stroke', d => edgeCfg[d.type] || '#94a3b8')
      .attr('stroke-width', 1.5)
      .attr('stroke-opacity', 0.7);

    const nodeEl = sel.append('g').selectAll('g')
      .data(egoNodes)
      .join('g')
      .attr('cursor', d => d.id === centerId ? 'default' : 'pointer')
      .on('click', (e, d) => {
        if (d.id !== centerId) window.location.href = `${d.id}.html`;
      });

    nodeEl.append('circle')
      .attr('r', d => d.id === centerId ? 10 : 7)
      .attr('fill', d => d.id === centerId ? '#2563eb' : '#7c3aed')
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5)
      .attr('fill-opacity', d => d.id === centerId ? 1 : 0.75);

    nodeEl.append('text')
      .attr('dy', d => d.id === centerId ? -13 : -10)
      .attr('text-anchor', 'middle')
      .attr('font-size', d => d.id === centerId ? 11 : 9)
      .attr('fill', d => d.id === centerId ? '#1e40af' : '#374151')
      .attr('font-weight', d => d.id === centerId ? '700' : '400')
      .text(d => {
        const label = d.id.replace(/^Skill-/, '').replace(/-/g, ' ');
        return label.length > 18 ? label.slice(0, 17) + '…' : label;
      });

    nodeEl.append('title').text(d => d.id);

    sim.on('tick', () => {
      linkEl
        .attr('x1', d => Math.max(10, Math.min(W - 10, d.source.x)))
        .attr('y1', d => Math.max(10, Math.min(H - 10, d.source.y)))
        .attr('x2', d => Math.max(10, Math.min(W - 10, d.target.x)))
        .attr('y2', d => Math.max(10, Math.min(H - 10, d.target.y)));
      nodeEl.attr('transform', d =>
        `translate(${Math.max(10, Math.min(W - 10, d.x))},${Math.max(12, Math.min(H - 8, d.y))})`
      );
    });

    sim.stop();
    for (let i = 0; i < 120; i++) sim.tick();
    sim.on('tick', () => {
      linkEl
        .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      nodeEl.attr('transform', d => `translate(${d.x},${d.y})`);
    });
    sim.restart();
  });
})();