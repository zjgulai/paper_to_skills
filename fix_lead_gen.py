import re
from pathlib import Path

def patch_lead_gen():
    p = Path('paper2skills-skills/playbook-generator/scripts/build_playbook.py')
    code = p.read_text()
    
    target_pattern = r"(    agent_cases_html = \(\n.*?\n    except Exception:\n        pass\n\n)(    body = f\"\"\")"
    
    # Elegant Gated Content
    injection = """
    # --- LEAD GEN GATE INJECTION ---
    code_section_html = ""
    if getattr(skill, 'is_alpha', False):
        code_section_html = f'''
        <!-- LEAD_GEN_GATE -->
        <div style="position: relative; margin-top: 40px; border-radius: var(--r-lg); border: 1px solid #E5E5E5; background: #FFFFFF; overflow: hidden;">
            <div style="filter: blur(6px); opacity: 0.3; padding: 32px; user-select: none; pointer-events: none;">
                <h3 style="margin-top:0; color: var(--ink);">③ 完整核心代码与实盘落地 SOP</h3>
                <pre style="background: #FAFAFA; border: 1px solid #F0F0F0; color: #999; padding: 16px; border-radius: 4px;"><code>import numpy as np\\nimport pandas as pd\\n\\ndef counterfactual_arbitrage(data):\\n    # Proprietary algorithms hidden...\\n    pass</code></pre>
            </div>
            
            <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; display: flex; flex-direction: column; justify-content: center; align-items: center; background: rgba(255,255,255,0.85); backdrop-filter: blur(2px);">
                <div style="width: 32px; height: 32px; border: 1px solid var(--ink); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-bottom: 16px;">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--ink)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect><path d="M7 11V7a5 5 0 0 1 10 0v4"></path></svg>
                </div>
                <h3 style="color: var(--ink); margin: 0 0 8px 0; font-size: 16px; font-weight: 600; letter-spacing: -0.3px;">解锁 Alpha 降维架构与闭环代码</h3>
                <p style="color: #666; margin: 0 0 24px 0; text-align: center; max-width: 360px; font-size: 13px; line-height: 1.6;">涉及核心商业机密映射，为保护跨域生态，仅对认证的高管开放。</p>
                
                <div style="display: flex; gap: 8px;">
                    <input type="email" placeholder="企业邮箱" style="padding: 0 16px; height: 36px; border: 1px solid #E5E5E5; border-radius: 4px; width: 220px; font-size: 13px; outline: none; background: #FAFAFA; transition: border-color 0.2s;">
                    <button onclick="alert('验证邮件已发送。\\n\\n【快速通道】请直接预约架构师 1v1 诊断。')" style="height: 36px; background: var(--ink); color: #fff; border: none; border-radius: 4px; padding: 0 20px; font-size: 13px; font-weight: 500; cursor: pointer; transition: opacity 0.2s;" onmouseover="this.style.opacity='0.8'" onmouseout="this.style.opacity='1'">认证并解锁</button>
                </div>
                <div style="margin-top: 16px; font-size: 12px; color: #999;">
                    或 <a href="mailto:skills@lute-tlz-dddd.top" style="color: var(--ink); font-weight: 500; text-decoration: none; border-bottom: 1px solid var(--ink);">预约 30 分钟私有化部署 Demo</a>
                </div>
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
    
    match = re.search(target_pattern, code, flags=re.DOTALL)
    if match:
        code = code[:match.start()] + match.group(1) + injection + match.group(2) + code[match.end():]
        
    render_skill_start = code.find('def render_skill_page')
    render_skill_end = code.find('def ', render_skill_start + 10)
    if render_skill_end == -1: render_skill_end = len(code)
    func_code = code[render_skill_start:render_skill_end]
    replaced_func = re.sub(r'<div class="skill-section">\s*<h2>💻 代码模板</h2>.*?</div>', '{code_section_html}', func_code, flags=re.DOTALL)
    code = code[:render_skill_start] + replaced_func + code[render_skill_end:]

    # Elegant DaaS Hook in Hero
    hero_injection = """  <div class="hero-primary-cta" style="margin-bottom: 32px;">
    <div style="margin-bottom: 24px; padding: 24px; background: #FFFFFF; border: 1px solid #E5E5E5; border-radius: var(--r-lg); display: inline-block; text-align: left; max-width: 580px;">
      <h3 style="margin: 0 0 6px 0; color: var(--ink); font-size: 15px; font-weight: 600; letter-spacing: -0.3px; display: flex; align-items: center; gap: 8px;">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--ink)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg>
        免费获取：竞品反事实体检报告 (DaaS)
      </h3>
      <p class="muted" style="margin: 0 0 16px 0; font-size: 13px; line-height: 1.5;">输入 Amazon ASIN，智能体集群将在 5 分钟内完成反事实诊断，精确定位利润流失与合规敞口。</p>
      <div style="display: flex; gap: 8px;">
        <input type="text" placeholder="输入 ASIN (如 B08F2...)" style="height: 36px; padding: 0 12px; border: 1px solid #E5E5E5; border-radius: 4px; flex-grow: 1; font-size: 13px; outline: none; background: #FAFAFA; transition: border-color 0.2s;">
        <button onclick="alert('报告生成任务已加入队列。\\n\\n检测到敏感竞品数据，报告已锁定。\\n请联系架构师获取专属解锁密钥。')" style="height: 36px; background: #F5F5F5; color: var(--ink); border: 1px solid #E5E5E5; border-radius: 4px; padding: 0 16px; font-size: 13px; font-weight: 500; cursor: pointer; white-space: nowrap; transition: background 0.2s;" onmouseover="this.style.background='#EFEFEF'" onmouseout="this.style.background='#F5F5F5'">生成法医报告</button>
      </div>
    </div>
    <br>
    <a class="btn-primary accent" href="ai-roadmap.html">获取私有化部署方案</a>
    <a class="btn-secondary" href="mailto:skills@lute-tlz-dddd.top?subject=预约诊断-paper2skills" >预约首席架构师诊断</a>"""
    
    code = re.sub(r'  <div class="hero-primary-cta">\s*<a class="btn-primary accent".*?</a>\s*</div>', hero_injection + '\n  </div>', code, flags=re.DOTALL)

    p.write_text(code)

patch_lead_gen()
print("Clean Lead Gen injected.")
