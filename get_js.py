with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'r', encoding='utf-8') as f:
    content = f.read()

import sys
sys.path.insert(0, 'paper2skills-skills/playbook-generator/scripts')
import build_playbook

js = build_playbook.build_chat_page_js()
print("JS length:", len(js))
print("matchRiskEvent in JS:", "matchRiskEvent" in js)
