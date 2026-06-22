import re
from pathlib import Path

def patch_playbook_skill():
    p = Path('paper2skills-skills/playbook-generator/scripts/build_playbook.py')
    code = p.read_text()
    
    if "is_alpha: bool" not in code:
        code = code.replace("    relations: dict[str, list[str]] = field(default_factory=dict)\n", 
                            "    relations: dict[str, list[str]] = field(default_factory=dict)\n    is_alpha: bool = False\n")
    
    if "is_alpha = any(" not in code:
        replacement = """        code_path_str = None
        is_alpha = any(k in text for k in ["反直觉", "非共识", "跨学科", "降维打击", "反事实", "流行病学", "高频交易", "拓扑数据", "数字孪生"])
"""
        code = code.replace("        code_path_str = None\n", replacement)
        
        # Add parameter to constructor
        code = code.replace("relations=relations,\n        )", "relations=relations,\n            is_alpha=is_alpha,\n        )")
    p.write_text(code)

def inject_alpha_dashboard():
    p = Path('paper2skills-skills/playbook-generator/scripts/build_playbook.py')
    code = p.read_text()
    
    if "alpha_skills =" not in code:
        alpha_extraction = """
    alpha_skills = [s for s in skills if s.is_alpha]
    alpha_cards = "".join(
        f"<a class='biz-card' style='border: 1px solid #B5323E;' href='skills/{s.skill_id}.html'>"
        f"<div class='biz-card-header'><span class='biz-icon' style='color:#B5323E;'>⚡</span>"
        f"<div class='biz-body'><div class='biz-card-meta'><strong>{html.escape(s.title)}</strong></div>"
        f"<p style='color:#B5323E; font-weight: 500;'>[Alpha 跨域降维打击]</p>"
        f"<p>{html.escape(s.problem_solved[:80])}...</p></div></div></a>"
        for s in alpha_skills[:12]
    )

    domain_cards = "".join(
"""
        code = code.replace("    domain_cards = \"\".join(\n", alpha_extraction)
        
        ceo_tab_replacement = """<div class="tab-panel" id="tab-ceo">
  <div style="background: #111; color: white; padding: 40px; border-radius: 12px; margin-bottom: 24px;">
    <h2 style="color: #fff; margin-top:0;">⚡ Alpha 级战略库：跨学科非共识雷达</h2>
    <p style="color: #999; font-size: 16px; max-width: 800px;">不要用同质化的工具去内卷。这里萃取了来自高频交易、流行病学、运筹学与拓扑学等跨学科顶刊算法，专门寻找母婴跨境赛道的「结构性漏洞」。</p>
    <div class="biz-grid" style="margin-top: 24px;">
      {alpha_cards}
    </div>
  </div>
  <h2>CEO 例外管理仪表盘 (Management by Exception)</h2>
  <div class="biz-grid">
    <div class="biz-card">
      <span style="color:#B5323E; font-size: 24px;">🚨</span>
      <h3>供应链断链预警</h3>
      <p>仅在检测到大宗商品期货异动导致未来 30 天 BOM 成本飙升时触发。</p>
    </div>
    <div class="biz-card">
      <span style="color:#0ea5e9; font-size: 24px;">📈</span>
      <h3>非共识套利战机</h3>
      <p>仅在汇率波动与竞品库存双重真空期时，触发「不降价保毛利」决策弹窗。</p>
    </div>
  </div>
</div>"""
        
        # Replace the empty or simple CEO tab
        code = re.sub(r'<div class="tab-panel" id="tab-ceo">.*?</div>', ceo_tab_replacement, code, flags=re.DOTALL)
        p.write_text(code)

patch_playbook_skill()
inject_alpha_dashboard()
print("Upgraded Playbook Architecture for Alpha Framework.")
