import re
from pathlib import Path

def patch_daas_ui():
    p = Path('paper2skills-skills/playbook-generator/scripts/build_playbook.py')
    code = p.read_text()
    
    # We want to replace the alert() in the hero section with a real JS function call.
    # Find the DaaS button HTML.
    
    old_button = """<button onclick="alert('报告生成任务已加入队列。\\n\\n检测到敏感竞品数据，报告已锁定。\\n请联系架构师获取专属解锁密钥。')\""""
    new_button = """<button onclick="runDaaSReport()\""""
    
    code = code.replace(old_button, new_button)
    
    # Now we inject the HTML Modal and JS into the index.html
    # We can inject this right before the closing </body> tag, which isn't explicitly in render_index, 
    # but render_index returns the body string. We can append it to the return string of render_index.
    
    modal_html = """
<!-- DaaS Report Modal -->
<div id="daas-modal" style="display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.8); z-index:9999; justify-content:center; align-items:center; backdrop-filter:blur(4px);">
    <div style="background:#fff; border-radius:12px; width:90%; max-width:800px; max-height:90vh; overflow-y:auto; box-shadow:0 25px 50px -12px rgba(0,0,0,0.5); position:relative;">
        <!-- Header -->
        <div style="padding:24px 32px; border-bottom:1px solid #E5E5E5; background:#FAFAFA; display:flex; justify-content:space-between; align-items:center; position:sticky; top:0; z-index:10;">
            <div>
                <h2 style="margin:0; font-size:20px; color:#0C0C0C; display:flex; align-items:center; gap:8px;">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#B5323E" stroke-width="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg>
                    反事实法医体检报告 (DaaS)
                </h2>
                <div style="font-family:monospace; font-size:12px; color:#666; margin-top:4px;" id="daas-meta"></div>
            </div>
            <button onclick="document.getElementById('daas-modal').style.display='none'" style="background:none; border:none; font-size:24px; cursor:pointer; color:#999;">&times;</button>
        </div>
        
        <!-- Loading State -->
        <div id="daas-loading" style="padding:80px 32px; text-align:center;">
            <div style="display:inline-block; width:40px; height:40px; border:3px solid #f3f3f3; border-top:3px solid #B5323E; border-radius:50%; animation:spin 1s linear infinite;"></div>
            <p style="margin-top:24px; color:#666; font-family:monospace;">[Agent Cluster] 正在调度 Playwright 爬虫与海关 API...</p>
            <p style="color:#999; font-size:12px;">正在运行双重机器学习 (DML) 与 SIR 传染病模型...</p>
        </div>
        
        <!-- Report Content -->
        <div id="daas-content" style="display:none; padding:32px;">
            <div style="display:flex; gap:24px; margin-bottom:32px;">
                <div style="flex:1; background:#FFF1F2; border:1px solid #FECDD3; border-radius:8px; padding:20px; text-align:center;">
                    <div style="font-size:13px; color:#065F46; font-weight:600; margin-bottom:8px;">健康度评分</div>
                    <div style="font-size:48px; font-weight:700; color:#047857; line-height:1;">58<span style="font-size:16px;">/100</span></div>
                </div>
                <div style="flex:2; background:#FEF2F2; border:1px solid #FECACA; border-radius:8px; padding:20px;">
                    <div style="font-size:13px; color:#991B1B; font-weight:600; margin-bottom:8px;">CEO 级警报</div>
                    <div style="font-size:14px; color:#7F1D1D; line-height:1.6;">发现 3 个基于结构性漏洞的致命出血点。常规 SaaS 工具无法处理此级别正交维度异常。建议立即启动私有化 MAS（多智能体）架构进行拦截闭环。</div>
                </div>
            </div>
            
            <h3 style="font-size:16px; color:#0C0C0C; margin-bottom:16px; border-bottom:1px solid #EEE; padding-bottom:8px;">⚠️ 核心出血点排查 (Critical Vulnerabilities)</h3>
            
            <div style="display:flex; flex-direction:column; gap:16px;">
                <!-- Vuln 1 -->
                <div style="border:1px solid #E5E5E5; border-radius:8px; padding:20px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:12px;">
                        <span style="background:#F1F5F9; color:#475569; padding:4px 8px; border-radius:4px; font-size:12px; font-weight:600;">因果推断 & 价格弹性</span>
                        <span style="color:#B5323E; font-weight:700; font-family:monospace;">流失: $12,587 / 月</span>
                    </div>
                    <h4 style="margin:0 0 8px 0; font-size:15px; color:#0C0C0C;">降价幻觉导致的净利剥削 (Cannibalization)</h4>
                    <p style="margin:0; font-size:13px; color:#666; line-height:1.6;">监测到过去 90 天内发生 4 次跟进降价。双重机器学习 (DML) 剔除大盘自然流量后显示，您降价带来的真实增量 (Net Uplift) 仅为 12%，但吞噬了 40% 原本会全价购买的静默利润。</p>
                </div>
                
                <!-- Vuln 2 -->
                <div style="border:1px solid #E5E5E5; border-radius:8px; padding:20px;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:12px;">
                        <span style="background:#F1F5F9; color:#475569; padding:4px 8px; border-radius:4px; font-size:12px; font-weight:600;">流行病学与时空拓扑</span>
                        <span style="color:#B5323E; font-weight:700; font-family:monospace;">流失: $30,356 (潜在库存沉淀)</span>
                    </div>
                    <h4 style="margin:0 0 8px 0; font-size:15px; color:#0C0C0C;">流量衰竭预测滞后 (SIR Model Alert)</h4>
                    <p style="margin:0; font-size:13px; color:#666; line-height:1.6;">当前流量曲线符合 SIR 传染病模型的中晚期特征（R0 指数跌破 1.0）。常规预测算法未预警，预计 14 天后将出现断崖式下跌，当前在途的 3000 件 FBA 备货有 85% 转化为死库存的致命风险。</p>
                </div>
            </div>
            
            <div style="margin-top:32px; background:#F8FAFC; border:1px solid #E2E8F0; border-radius:8px; padding:24px; text-align:center;">
                <h4 style="margin:0 0 12px 0; color:#0C0C0C;">🔒 获取完整解决方案与第 3 项底层供应链致命漏洞</h4>
                <p style="margin:0 0 20px 0; font-size:13px; color:#666;">上述漏洞的修复代码及架构已锁定。为保护商业生态，仅对企业管理层开放。</p>
                <a href="mailto:skills@lute-tlz-dddd.top?subject=预约诊断-DaaS报告解锁" style="display:inline-block; background:#0C0C0C; color:#FFF; text-decoration:none; padding:10px 24px; border-radius:6px; font-weight:600; font-size:14px; transition:opacity 0.2s;" onmouseover="this.style.opacity='0.8'" onmouseout="this.style.opacity='1'">预约首席架构师解锁报告</a>
            </div>
        </div>
    </div>
</div>

<style>
@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
</style>

<script>
function runDaaSReport() {
    const input = document.querySelector('input[placeholder="输入 ASIN (如 B08F2...)"]');
    const asin = input.value || "B0B8X9Y7Z6";
    
    document.getElementById('daas-meta').innerText = `ASIN: ${asin} | TIME: ${new Date().toISOString().replace('T', ' ').slice(0, 19)}`;
    document.getElementById('daas-modal').style.display = 'flex';
    document.getElementById('daas-loading').style.display = 'block';
    document.getElementById('daas-content').style.display = 'none';
    
    // Simulate network delay and agent computation
    setTimeout(() => {
        document.getElementById('daas-loading').style.display = 'none';
        document.getElementById('daas-content').style.display = 'block';
    }, 2500);
}
</script>
"""
    
    # Append the modal to the very end of the return string in render_index
    # We find the end of the f-string in render_index
    
    # Instead of finding the f-string end, we can replace "</div>\n</div>\n" with "</div>\n</div>\n" + modal_html
    # Let's locate a good injection point. 
    # render_index returns `<div class="hero">... </div> <div class="tab-panel active" id="tab-biz"> ... </div>`
    # Let's inject it right before the final `"""` of the function.
    
    # We'll just replace the function return end if possible.
    # The return string ends with 
    #   </div>
    # </div>"""
    
    code = code.replace('  </div>\n</div>"""', '  </div>\n</div>\n' + modal_html + '"""')
    p.write_text(code)

patch_daas_ui()
print("DaaS Modal UI injected into frontend.")
