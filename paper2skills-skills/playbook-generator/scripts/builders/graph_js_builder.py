"""Graph JS asset builders for paper2skills Playbook."""
from __future__ import annotations


def build_ego_graph_js() -> str:
    """Ego graph for Skill detail pages: renders 1-hop neighbourhood in the relation panel."""
    return r"""
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
""".strip()


def build_graph_js() -> str:
    """Return the D3 force graph JS bundle."""
    return r"""
document.addEventListener('DOMContentLoaded', function () {
(function () {
  const DATA = window.PLAYBOOK_DATA || {};
  const skills = DATA.skills || [];
  const skillMap = {};
  skills.forEach(s => { skillMap[s.skill_id] = s; });

  // Load graph-data.json via XHR (works for file:// and http://)
  function loadGraphData(cb) {
    const xhr = new XMLHttpRequest();
    xhr.open('GET', '../assets/graph-data.json');
    xhr.onload = () => {
      try { cb(JSON.parse(xhr.responseText)); } catch (e) { cb(null); }
    };
    xhr.onerror = () => cb(null);
    xhr.send();
  }

  loadGraphData(function (raw) {
    if (!raw) { document.getElementById('graph-svg').insertAdjacentHTML('beforebegin', '<p class="muted">无法加载图谱数据。</p>'); return; }

    const nodes = raw.nodes.map(n => ({ ...n }));
    const nodeIdSet = new Set(nodes.map(n => n.id));
    const links = raw.links
      .map(l => ({ ...l }))
      .filter(l => nodeIdSet.has(l.source) && nodeIdSet.has(l.target));

    // Domain colour palette (Tableau-10 extended)
    const domains = [...new Set(nodes.map(n => n.domain))].sort();
    const colour = d3.scaleOrdinal(d3.schemeTableau10.concat(d3.schemePastel1)).domain(domains);

    // Degree map for node sizing
    const degree = {};
    links.forEach(l => {
      degree[l.source] = (degree[l.source] || 0) + 1;
      degree[l.target] = (degree[l.target] || 0) + 1;
    });
    const maxDeg = Math.max(...Object.values(degree), 1);
    const rScale = d3.scaleSqrt().domain([0, maxDeg]).range([4, 14]);

    const svg = d3.select('#graph-svg');
    const W = svg.node().parentElement.clientWidth || 1100;
    const H = Math.max(600, window.innerHeight - 240);
    svg.attr('width', W).attr('height', H).attr('viewBox', `0 0 ${W} ${H}`);

    const g = svg.append('g');

    // Zoom
    svg.call(d3.zoom().scaleExtent([0.1, 6]).on('zoom', e => g.attr('transform', e.transform)));

    // Edge type → display config
    const edgeCfg = {
      prerequisite: { stroke: '#3b82f6', dasharray: null, width: 1.5 },
      combinable:   { stroke: '#10b981', dasharray: '5,3',   width: 1 },
      extension:    { stroke: '#f59e0b', dasharray: '2,4',   width: 1 },
    };

    // Visibility state
    const visible = { prerequisite: true, combinable: true, extension: false };
    document.querySelectorAll('.graph-controls input[type=checkbox]').forEach(cb => {
      cb.addEventListener('change', () => {
        visible[cb.id.replace('cb-', '')] = cb.checked;
        updateEdgeVisibility();
      });
    });

    function updateEdgeVisibility() {
      linkEl.style('display', d => visible[d.type] ? null : 'none');
    }

    // Simulation — only prerequisite + combinable edges by default for perf
    const activeLinks = links.filter(l => l.type === 'prerequisite' || l.type === 'combinable');
    const sim = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(activeLinks).id(d => d.id).distance(60).strength(0.4))
      .force('charge', d3.forceManyBody().strength(-80))
      .force('center', d3.forceCenter(W / 2, H / 2))
      .force('collide', d3.forceCollide().radius(d => rScale(degree[d.id] || 0) + 4));

    // Draw all edges (extension hidden initially)
    const linkEl = g.append('g').selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', d => (edgeCfg[d.type] || edgeCfg.prerequisite).stroke)
      .attr('stroke-width', d => (edgeCfg[d.type] || edgeCfg.prerequisite).width)
      .attr('stroke-dasharray', d => (edgeCfg[d.type] || edgeCfg.prerequisite).dasharray)
      .attr('stroke-opacity', 0.5)
      .style('display', d => visible[d.type] ? null : 'none');

    // Draw nodes
    const nodeEl = g.append('g').selectAll('circle')
      .data(nodes)
      .join('circle')
      .attr('r', d => rScale(degree[d.id] || 0))
      .attr('fill', d => colour(d.domain))
      .attr('stroke', '#fff')
      .attr('stroke-width', 1.5)
      .attr('cursor', 'pointer')
      .call(d3.drag()
        .on('start', (e, d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
        .on('drag',  (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on('end',   (e, d) => { if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; })
      );

    // Hover: highlight 1-hop neighbourhood
    const neighborSet = new Set();
    nodeEl
      .on('mouseover', (e, d) => {
        neighborSet.clear();
        neighborSet.add(d.id);
        links.forEach(l => {
          const src = typeof l.source === 'object' ? l.source.id : l.source;
          const tgt = typeof l.target === 'object' ? l.target.id : l.target;
          if (src === d.id || tgt === d.id) { neighborSet.add(src); neighborSet.add(tgt); }
        });
        nodeEl.attr('opacity', n => neighborSet.has(n.id) ? 1 : 0.15);
        linkEl.attr('stroke-opacity', l => {
          const src = typeof l.source === 'object' ? l.source.id : l.source;
          const tgt = typeof l.target === 'object' ? l.target.id : l.target;
          return (neighborSet.has(src) && neighborSet.has(tgt)) ? 0.8 : 0.05;
        });
      })
      .on('mouseout', () => {
        nodeEl.attr('opacity', 1);
        linkEl.attr('stroke-opacity', 0.5);
      })
      .on('click', (e, d) => showInfo(d));

    // Tick
    sim.on('tick', () => {
      linkEl
        .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
      nodeEl.attr('cx', d => d.x).attr('cy', d => d.y);
    });

    // Info panel
    const infoPanel = document.getElementById('graph-info');
    document.getElementById('graph-info-close').addEventListener('click', () => infoPanel.classList.add('hidden'));

    function showInfo(d) {
      const sk = skillMap[d.id];
      document.getElementById('gi-title').textContent = sk ? sk.title : d.id;
      document.getElementById('gi-domain').textContent = d.domain || '';
      document.getElementById('gi-summary').textContent = sk ? (sk.problem_solved || sk.algorithm_summary || '') : '';
      const link = document.getElementById('gi-link');
      link.href = `../skills/${d.id}.html`;
      infoPanel.classList.remove('hidden');
    }

    // Search
    document.getElementById('graph-search').addEventListener('input', function () {
      const q = this.value.trim().toLowerCase();
      if (!q) { nodeEl.attr('opacity', 1); return; }
      nodeEl.attr('opacity', d => (d.id.toLowerCase().includes(q) || (skillMap[d.id] && skillMap[d.id].title.toLowerCase().includes(q))) ? 1 : 0.1);
    });

    updateEdgeVisibility();
  });
})();
});
"""


# ---------------------------------------------------------------------------
# CSS + JS assets (Phase 3C/D additions merged)

