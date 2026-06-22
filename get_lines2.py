import re
with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Let's see what function contains this definition
idx = content.find("const matchedEvt=(function(){if(!window.RISK_EVENTS||!window.RISK_EVENTS.events)return null;")
print("Found inside block starting at:", content.rfind("def ", 0, idx))
print("The function name is:", content[content.rfind("def ", 0, idx):content.find("\n", content.rfind("def ", 0, idx))])

