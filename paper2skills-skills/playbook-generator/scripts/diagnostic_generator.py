import re

def insert_diagnostic_page(build_py_path):
    with open(build_py_path, 'r', encoding='utf-8') as f:
        content = f.read()

    diagnostic_page_code = """
def render_diagnostic_page(nav: str = "", skill_count: int = 849) -> str:
    body = f'''
    <div class="chat-layout">
      <!-- 顶栏 -->
      <div class="chat-topbar">
        <a href="{nav}index.html" class="chat-back">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 12H5M12 19l-7-7 7-7"/></svg>
          返回
        </a>
        <div class="chat-title-area">
          <span class="chat-title-icon">🩺</span>
          <span class="chat-title-text">业务诊断中心</span>
          <span class="chat-title-sub">AI Diagnostic</span>
        </div>
      </div>

      <!-- 诊断主体区域 -->
      <div class="diag-container">
        <!-- 左侧症状选择区 -->
        <div class="diag-sidebar">
          <div class="diag-header">
            <h3>常见业务症状</h3>
            <p>点击下方常见症状快速诊断，或在输入框描述你的具体问题。</p>
          </div>
          
          <div class="diag-symptoms" id="symptoms-container">
            <!-- 症状按钮由 JS 动态生成 -->
          </div>

          <div class="diag-input-area">
            <textarea id="symptom-input" placeholder="例如：最近我们的一款核心ASIN自然流量突然下降了30%，同时转化率也跌了..." rows="4"></textarea>
            <button id="btn-diagnose" class="btn-primary">开始诊断 🪄</button>
          </div>
        </div>

        <!-- 右侧诊断结果区 -->
        <div class="diag-main">
          <div id="diag-empty" class="diag-empty-state">
            <div class="empty-icon">🏥</div>
            <h3>等待诊断</h3>
            <p>描述你遇到的业务问题，AI 将为你匹配最相关的诊断 Skill 链，<br>提供从根因排查到长效预防的完整解决方案。</p>
          </div>
          
          <div id="diag-result" class="diag-result-area" style="display: none;">
            <!-- 诊断结果由 JS 动态生成 -->
          </div>
        </div>
      </div>
    </div>

    <!-- 内联 CSS -->
    <style>
      :root {{
        --diag-border: #E5E7EB;
        --diag-panel: #FFFFFF;
        --diag-hover: #F9FAFB;
      }}
      body {{ overflow: hidden; background: var(--bg); }}
      
      .diag-container {{
        flex: 1;
        display: flex;
        overflow: hidden;
      }}
      
      /* 左侧侧边栏 */
      .diag-sidebar {{
        width: 380px;
        background: var(--diag-panel);
        border-right: 1px solid var(--diag-border);
        display: flex;
        flex-direction: column;
        flex-shrink: 0;
        box-shadow: 2px 0 8px rgba(0,0,0,0.02);
        z-index: 10;
      }}
      
      .diag-header {{
        padding: 24px 20px 16px;
        border-bottom: 1px solid var(--diag-border);
      }}
      
      .diag-header h3 {{
        margin: 0 0 8px;
        font-size: 16px;
        font-weight: 600;
        color: var(--ink);
      }}
      
      .diag-header p {{
        margin: 0;
        font-size: 13px;
        color: var(--muted);
        line-height: 1.5;
      }}
      
      .diag-symptoms {{
        flex: 1;
        overflow-y: auto;
        padding: 16px 20px;
        display: flex;
        flex-direction: column;
        gap: 8px;
      }}
      
      .symptom-btn {{
        display: flex;
        align-items: center;
        gap: 10px;
        width: 100%;
        padding: 12px 14px;
        border: 1px solid var(--diag-border);
        background: #fff;
        border-radius: var(--r-md);
        font-size: 14px;
        color: var(--ink);
        cursor: pointer;
        text-align: left;
        transition: all var(--t);
        font-family: inherit;
      }}
      
      .symptom-btn:hover {{
        border-color: var(--accent);
        background: var(--accent-light);
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(181, 50, 62, 0.05);
      }}
      
      .symptom-icon {{
        font-size: 18px;
      }}
      
      .diag-input-area {{
        padding: 20px;
        border-top: 1px solid var(--diag-border);
        background: var(--diag-panel);
      }}
      
      .diag-input-area textarea {{
        width: 100%;
        padding: 12px;
        border: 1px solid var(--diag-border);
        border-radius: var(--r-md);
        resize: none;
        font-family: inherit;
        font-size: 14px;
        line-height: 1.5;
        margin-bottom: 12px;
        transition: border-color var(--t);
      }}
      
      .diag-input-area textarea:focus {{
        outline: none;
        border-color: var(--accent);
      }}
      
      .btn-primary {{
        width: 100%;
        padding: 12px;
        background: var(--accent);
        color: #fff;
        border: none;
        border-radius: var(--r-md);
        font-size: 14px;
        font-weight: 600;
        cursor: pointer;
        transition: opacity var(--t);
      }}
      
      .btn-primary:hover {{
        opacity: 0.9;
      }}
      
      /* 右侧主要区域 */
      .diag-main {{
        flex: 1;
        overflow-y: auto;
        padding: 40px;
        background: var(--bg);
      }}
      
      .diag-empty-state {{
        height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        color: var(--muted);
      }}
      
      .empty-icon {{
        font-size: 48px;
        margin-bottom: 16px;
        opacity: 0.5;
      }}
      
      .diag-empty-state h3 {{
        margin: 0 0 12px;
        color: var(--ink);
        font-size: 18px;
      }}
      
      .diag-empty-state p {{
        font-size: 14px;
        line-height: 1.6;
        max-width: 400px;
      }}
      
      .diag-result-area {{
        max-width: 800px;
        margin: 0 auto;
      }}
      
      .diag-event-header {{
        background: #fff;
        padding: 24px;
        border-radius: var(--r-lg);
        border: 1px solid var(--diag-border);
        margin-bottom: 32px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
      }}
      
      .event-title-row {{
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 12px;
      }}
      
      .event-title {{
        margin: 0;
        font-size: 20px;
        font-weight: 600;
        color: var(--ink);
      }}
      
      .severity-badge {{
        padding: 4px 10px;
        border-radius: var(--r-full);
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
      }}
      
      .sev-critical {{ background: #FEF2F2; color: #DC2626; border: 1px solid #F87171; }}
      .sev-high {{ background: #FFFBEB; color: #D97706; border: 1px solid #FBBF24; }}
      .sev-medium {{ background: #EFF6FF; color: #2563EB; border: 1px solid #60A5FA; }}
      .sev-low {{ background: #F0FDF4; color: #16A34A; border: 1px solid #4ADE80; }}
      
      .event-summary {{
        margin: 0;
        font-size: 14px;
        color: var(--muted);
        line-height: 1.5;
      }}
      
      /* Phase 容器 */
      .phase-section {{
        margin-bottom: 32px;
      }}
      
      .phase-title {{
        margin: 0 0 16px;
        font-size: 16px;
        font-weight: 600;
        color: var(--ink);
        display: flex;
        align-items: center;
        gap: 8px;
      }}
      
      .phase-cards {{
        display: flex;
        flex-direction: column;
        gap: 12px;
      }}
      
      /* Skill 条目卡片 */
      .skill-item-card {{
        background: #fff;
        border: 1px solid var(--diag-border);
        border-radius: var(--r-md);
        padding: 16px;
        display: flex;
        flex-direction: column;
        gap: 8px;
        transition: transform .22s cubic-bezier(0.4,0,0.2,1), box-shadow .22s cubic-bezier(0.4,0,0.2,1), border-color .15s ease;
      }}
      
      .skill-item-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.06);
        border-color: var(--accent);
      }}
      
      .skill-header {{
        display: flex;
        align-items: flex-start;
        gap: 12px;
      }}
      
      .skill-seq {{
        display: flex;
        align-items: center;
        justify-content: center;
        width: 24px;
        height: 24px;
        background: var(--bg);
        color: var(--muted);
        font-size: 12px;
        font-weight: 600;
        border-radius: 50%;
        flex-shrink: 0;
      }}
      
      .skill-link {{
        font-size: 15px;
        font-weight: 600;
        color: var(--accent);
        text-decoration: none;
        flex: 1;
        line-height: 1.4;
      }}
      
      .skill-link:hover {{
        text-decoration: underline;
      }}
      
      .skill-role {{
        margin: 0;
        font-size: 14px;
        color: var(--ink);
        line-height: 1.5;
        padding-left: 36px;
      }}
      
      .skill-condition {{
        margin-left: 36px;
        margin-top: 4px;
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        background: #F3F4F6;
        border-radius: var(--r-sm);
        font-size: 12px;
        color: #4B5563;
        border: 1px dashed #D1D5DB;
        width: fit-content;
      }}
      
      .condition-label {{
        font-weight: 600;
        margin-right: 4px;
      }}
      
      @media (max-width: 768px) {{
        .diag-container {{ flex-direction: column; overflow: auto; }}
        .diag-sidebar {{ width: 100%; border-right: none; border-bottom: 1px solid var(--diag-border); }}
        .diag-symptoms {{ max-height: 200px; flex-direction: row; flex-wrap: wrap; overflow-x: hidden; }}
        .symptom-btn {{ width: calc(50% - 4px); padding: 8px 10px; }}
        .diag-main {{ padding: 20px; overflow: visible; }}
      }}
    </style>

    <!-- JS 逻辑 -->
    <script src="{nav}assets/playbook-data.js"></script>
    <script src="{nav}assets/risk-events.js"></script>
    <script>
      document.addEventListener('DOMContentLoaded', () => {{
        const events = window.RISK_EVENTS?.events || [];
        const symptomsContainer = document.getElementById('symptoms-container');
        const diagInput = document.getElementById('symptom-input');
        const btnDiagnose = document.getElementById('btn-diagnose');
        const diagEmpty = document.getElementById('diag-empty');
        const diagResult = document.getElementById('diag-result');
        
        // 1. 渲染症状快捷按钮
        if (events.length > 0) {{
          symptomsContainer.innerHTML = '';
          events.forEach(ev => {{
            const btn = document.createElement('button');
            btn.className = 'symptom-btn';
            btn.innerHTML = `<span class="symptom-icon">${{ev.icon}}</span><span>${{ev.event_name}}</span>`;
            btn.onclick = () => {{
              diagInput.value = ev.symptom_keywords.slice(0, 3).join('，') + '...';
              renderResult(ev);
            }};
            symptomsContainer.appendChild(btn);
          }});
        }}
        
        // 2. 诊断匹配逻辑
        function runDiagnosis() {{
          const text = diagInput.value.trim().toLowerCase();
          if (!text) return;
          
          let bestMatch = null;
          let maxScore = 0;
          
          events.forEach(ev => {{
            let score = 0;
            // 简单的关键词匹配评分
            ev.symptom_keywords.forEach(kw => {{
              if (text.includes(kw.toLowerCase())) {{
                score += kw.length; // 越长的词权重大
              }}
            }});
            
            if (score > maxScore) {{
              maxScore = score;
              bestMatch = ev;
            }}
          }});
          
          if (bestMatch) {{
            renderResult(bestMatch);
          }} else {{
            // 如果没有匹配，默认选第一个或给提示（这里选得分最高或提示）
            if (events.length > 0) renderResult(events[0]);
          }}
        }}
        
        btnDiagnose.onclick = runDiagnosis;
        
        // 3. 渲染结果
        function renderResult(ev) {{
          diagEmpty.style.display = 'none';
          diagResult.style.display = 'block';
          
          const sevClass = `sev-${{ev.severity.toLowerCase()}}`;
          
          let html = `
            <div class="diag-event-header">
              <div class="event-title-row">
                <span style="font-size: 24px;">${{ev.icon}}</span>
                <h2 class="event-title">${{ev.event_name}}</h2>
                <span class="severity-badge ${{sevClass}}">${{ev.severity}}</span>
              </div>
              <p class="event-summary">${{ev.summary}}</p>
            </div>
          `;
          
          const phaseConfig = [
            {{ key: 'diagnose', title: '🔍 第一步：诊断根因' }},
            {{ key: 'treat', title: '🔧 第二步：处置行动' }},
            {{ key: 'prevent', title: '🛡️ 第三步：长效预防' }}
          ];
          
          phaseConfig.forEach(pc => {{
            const skills = ev.phases[pc.key] || [];
            if (skills.length === 0) return;
            
            let cardsHtml = skills.map((sk, idx) => {{
              const condHtml = sk.condition ? 
                `<div class="skill-condition"><span class="condition-label">触发条件：</span>${{sk.condition}}</div>` : '';
              
              return `
                <div class="skill-item-card">
                  <div class="skill-header">
                    <span class="skill-seq">${{idx + 1}}</span>
                    <a href="{nav}skills/${{sk.skill_id}}.html" class="skill-link" target="_blank">
                      ${{sk.skill_id}}
                    </a>
                  </div>
                  <p class="skill-role">${{sk.role}}</p>
                  ${{condHtml}}
                </div>
              `;
            }}).join('');
            
            html += `
              <div class="phase-section">
                <h3 class="phase-title">${{pc.title}}</h3>
                <div class="phase-cards">
                  ${{cardsHtml}}
                </div>
              </div>
            `;
          }});
          
          diagResult.innerHTML = html;
        }}
      }});
    </script>
    '''
    return html_page("业务诊断中心 · paper2skills", body, nav=nav, active_nav="diagnostic")
"""

    if 'def render_diagnostic_page(' not in content:
        # Find def render_chat_page and insert before it
        content = content.replace('def render_chat_page(', diagnostic_page_code + '\ndef render_chat_page(')
        print("Inserted render_diagnostic_page")
        
        # Add write_file line
        content = content.replace('write_file(out / "chat.html", render_chat_page(skill_count=skill_count))', 
                                  'write_file(out / "chat.html", render_chat_page(skill_count=skill_count))\n    write_file(out / "diagnostic.html", render_diagnostic_page(skill_count=skill_count))')
        print("Inserted write_file for diagnostic.html")
        
        with open(build_py_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("File updated")
    else:
        print("Already exists")

insert_diagnostic_page('paper2skills-skills/playbook-generator/scripts/build_playbook.py')
