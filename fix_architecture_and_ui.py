import re
from pathlib import Path

def patch_system():
    p = Path('paper2skills-skills/playbook-generator/scripts/build_playbook.py')
    code = p.read_text()
    
    # 1. Add `is_alpha: bool` to PlaybookSkill class
    if "is_alpha: bool" not in code:
        code = code.replace("    relations: dict[str, list[str]] = field(default_factory=dict)\n", 
                            "    relations: dict[str, list[str]] = field(default_factory=dict)\n    is_alpha: bool = False\n")
    
    # 2. Add is_alpha identification logic
    if "is_alpha = any(" not in code:
        replacement = """        code_path_str = None
        is_alpha = any(k in text for k in ["反直觉", "非共识", "跨学科", "降维打击", "反事实", "流行病学", "高频交易", "拓扑数据", "数字孪生"])
"""
        code = code.replace("        code_path_str = None\n", replacement)
        code = code.replace("relations=relations,\n        )", "relations=relations,\n            is_alpha=is_alpha,\n        )")

    # 3. Create elegant Alpha cards HTML
    # Rules: No default shadow, 1px border, 8px radius, restrained colors.
    alpha_extraction = """
    alpha_skills = [s for s in skills if s.is_alpha]
    alpha_cards = "".join(
        f"<a class='biz-card' href='skills/{s.skill_id}.html' style='border-color: var(--accent);'>"
        f"<div class='biz-card-header'>"
        f"<span class='biz-icon' style='color: var(--accent); background: transparent; border: 1px solid var(--accent);'>⚡</span>"
        f"<div class='biz-body'><div class='biz-card-meta'><strong style='color: var(--ink);'>{html.escape(s.title)}</strong></div>"
        f"<p style='color: var(--accent); font-weight: 500; font-size: 12px; margin-bottom: 4px;'>[ Alpha 跨域非共识 ]</p>"
        f"<p>{html.escape(s.problem_solved[:80])}...</p></div></div></a>"
        for s in alpha_skills[:12]
    )

    domain_cards = "".join(
"""
    code = code.replace("    domain_cards = \"\".join(\n", alpha_extraction)
    
    # 4. Redesign the CEO tab with Linear / Smartisan aesthetic
    ceo_tab_replacement = """<div class="tab-panel" id="tab-ceo">
  <div style="margin-bottom: 32px; padding: 32px; border: 1px solid #E5E5E5; border-radius: var(--r-lg); background: #FFFFFF;">
    <h2 style="color: var(--ink); margin-top:0; font-size: 20px; font-weight: 600; letter-spacing: -0.5px;">⚡ Alpha 战略库 (Cross-Domain Alpha)</h2>
    <p class="muted" style="max-width: 800px; margin-bottom: 24px; line-height: 1.6;">拒绝同质化工具内卷。萃取高频交易、流行病学与拓扑学等跨学科算法，探寻母婴跨境赛道的结构性漏洞与非共识套利空间。</p>
    <div class="biz-grid">
      {alpha_cards}
    </div>
  </div>
  
  <h2 style="color: var(--ink); font-size: 20px; font-weight: 600; letter-spacing: -0.5px;">例外管理引擎 (Management by Exception)</h2>
  <div class="biz-grid">
    <div class="biz-card" style="border-radius: var(--r-lg);">
      <div style="color: var(--ink); font-size: 14px; font-weight: 600; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;">
        <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: var(--accent);"></span>
        断链预警
      </div>
      <p class="muted" style="margin: 0; font-size: 13px;">检测到大宗商品异动导致 BOM 成本飙升时触发。</p>
    </div>
    <div class="biz-card" style="border-radius: var(--r-lg);">
      <div style="color: var(--ink); font-size: 14px; font-weight: 600; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;">
        <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #059669;"></span>
        非共识套利
      </div>
      <p class="muted" style="margin: 0; font-size: 13px;">汇率波动与竞品库存双重真空期时，触发阻断降价决策。</p>
    </div>
  </div>
</div>"""
    
    code = re.sub(r'<div class="tab-panel" id="tab-ceo">.*?</div>', ceo_tab_replacement, code, flags=re.DOTALL)
    
    p.write_text(code)

patch_system()
print("Base Architecture Updated (Clean Design)")
