import json

with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'r', encoding='utf-8') as f:
    content = f.read()

# I want to find the exact definition of build_chat_page_js and replace it.
print("Function length:", len(content.split("def build_chat_page_js() -> str:")[1]))
