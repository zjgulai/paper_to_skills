with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'r', encoding='utf-8') as f:
    content = f.read()

import sys
sys.path.insert(0, 'paper2skills-skills/playbook-generator/scripts')
import build_playbook

js = build_playbook.build_chat_page_js()

with open('playbook/assets/chat-page.js', 'r', encoding='utf-8') as f:
    output_js = f.read()
    
print("Lengths:", len(js), len(output_js))
