with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

# Is there any build_chat_page_js defined later that overwrites it?
matches = [m.start() for m in re.finditer(r'def build_chat_page_js', content)]
print("Indices of build_chat_page_js definitions:", matches)

