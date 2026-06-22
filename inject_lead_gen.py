import re
from pathlib import Path

def patch_render_skill_page():
    p = Path('paper2skills-skills/playbook-generator/scripts/build_playbook.py')
    code = p.read_text()
    
    # 1. Inject logic into render_skill_page
    # We will locate the end of `render_skill_page` function body before `body = f"""`
    
    target_pattern = r"(    agent_cases_html = \(\n.*?\n    except Exception:\n        pass\n\n)(    body = f\"\"\")"
    
    injection = """
    # --- LEAD GEN GATE INJECTION ---
    code_section_html = ""
    if getattr(skill, 'is_alpha', False):
        code_section_html = f'''
        <!-- LEAD_GEN_GATE -->
        <div style="position: relative; margin-top: 32px; border-radius: 12px; overflow: hidden; border: 1px solid #e2e8f0; background: #f8fafc;">
            <div style="filter: blur(8px); opacity: 0.4; padding: 24px; user-select: none; pointer-events: none;">
                <h3 style="margin-top:0;">③ 完整核心代码与实盘落地 SOP</h3>
                <pre style="background: #1e293b; color: #e2e8f0; padding: 16px; border-radius: 8px;"><code>import numpy as np\\nimport pandas as pd\\n\\ndef counterfactual_arbitrage(data):\\n    # Secret algorithm running...\\n    return ultra_profit</code></pre>
            </div>
            <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; display: flex; flex-direction: column; justify-content: center; align-items: center; background: rgba(255,255,255,0.7); backdrop-filter: blur(2px);">
                <span style="font-size: 32px; margin-bottom: 12px;">🔒</span>
                <h3 style="color: #111; margin: 0 0 8px 0; font-size: 20px;">解锁 Alpha 级降维代码与 SOP</h3>
                <p style="color: #475569; margin: 0 0 20px 0; text-align: center; max-width: 400px; font-size: 14px;">该反直觉算法涉及核心商业机密架构，为保护生态健康，仅对认证的品牌决策者开放。</p>
                <div style="display: flex; gap: 12px;">
                    <input type="email" placeholder="输入企业邮箱验证" style="padding: 10px 16px; border: 1px solid #cbd5e1; border-radius: 6px; width: 240px; font-size: 14px; outline: none;">
                    <button onclick="alert('验证邮件已发送，请检查收件箱。\\n\\n【商务通道】也可直接预约首席架构师 1v1 诊断。')" style="background: #B5323E; color: #fff; border: none; border-radius: 6px; padding: 0 20px; font-weight: 600; cursor: pointer;">立即解锁</button>
                </div>
                <p style="margin-top: 16px; font-size: 12px; color: #64748b;">或 <a href="mailto:skills@lute-tlz-dddd.top" style="color: #B5323E; text-decoration: underline;">预约 30 分钟私有化部署 Demo</a></p>
            </div>
        </div>
        '''
    else:
        code_html = f"<div class='code-preview'>{skill.code_preview}</div>" if skill.code_preview else ""
        repo_link = f"<div class='repo-link'><a href='https://github.com/your-org/paper2skills/tree/main/{skill.code_path}' target='_blank'>在 GitHub 查看完整代码 ↗</a></div>" if skill.code_path else ""
        code_section_html = f'''
        <div class="skill-section">
            <h2>💻 代码模板</h2>
            {code_html}
            {repo_link}
        </div>
        '''
    # --- END INJECTION ---
"""
    
    # Replace the matching part
    match = re.search(target_pattern, code, flags=re.DOTALL)
    if match:
        code = code[:match.start()] + match.group(1) + injection + match.group(2) + code[match.end():]
    
    # Now replace the raw code section rendering inside render_skill_page's f-string return.
    # We find `<div class="skill-section">\n  <h2>💻 代码模板</h2>\n  {code_html}\n  {repo_link}\n</div>` inside render_skill_page.
    
    render_skill_start = code.find('def render_skill_page')
    render_skill_end = code.find('def ', render_skill_start + 10)
    if render_skill_end == -1:
        render_skill_end = len(code)
        
    func_code = code[render_skill_start:render_skill_end]
    
    replaced_func = re.sub(
        r'<div class="skill-section">\s*<h2>💻 代码模板</h2>.*?</div>', 
        '{code_section_html}', 
        func_code, 
        flags=re.DOTALL
    )
    
    code = code[:render_skill_start] + replaced_func + code[render_skill_end:]

    # 2. Inject DaaS into render_index
    hero_injection = """  <div class="hero-primary-cta">
    <div style="margin-bottom: 24px; padding: 24px; background: rgba(255,255,255,0.95); border: 1px solid #e2e8f0; border-radius: 12px; display: inline-block; box-shadow: 0 4px 12px -2px rgba(0,0,0,0.1); text-align: left; max-width: 600px;">
      <h3 style="margin: 0 0 8px 0; color: #0f172a; font-size: 18px;">⚕️ 免费：获取竞品反事实体检报告 (DaaS)</h3>
      <p style="margin: 0 0 16px 0; color: #475569; font-size: 14px;">输入您或竞品的 Amazon ASIN，我们的分析 Agent 将在 5 分钟内发回体检报告，精准定位利润流失点与合规风险。</p>
      <div style="display: flex; gap: 8px;">
        <input type="text" placeholder="输入 ASIN (如 B08F2...)" style="padding: 12px 16px; border: 1px solid #cbd5e1; border-radius: 6px; flex-grow: 1; font-size: 14px;">
        <button onclick="alert('爬虫集群已启动！\\n\\n检测到高价值竞争数据，为防止滥用，报告下载密码已加密。\\n请点击下方「预约 30 分钟 Demo」联系架构师获取。')" style="background: #0ea5e9; color: #fff; border: none; border-radius: 6px; padding: 0 24px; font-weight: 600; cursor: pointer; white-space: nowrap;">生成法医级报告</button>
      </div>
    </div>
    <br>
    <a class="btn-primary accent" href="ai-roadmap.html">查看私有化部署方案</a>
    <a class="btn-secondary" href="mailto:skills@lute-tlz-dddd.top?subject=预约Demo-paper2skills" >预约 30 分钟 Demo</a>"""
    
    code = re.sub(r'  <div class="hero-primary-cta">\s*<a class="btn-primary accent".*?</a>\s*</div>', hero_injection + '\n  </div>', code, flags=re.DOTALL)

    p.write_text(code)

patch_render_skill_page()
print("Lead Gen and Gated Content features injected carefully.")
