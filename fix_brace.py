with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace("    }}}\n    .chat-ctrl {{", "    }}\n    .chat-ctrl {{")

with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'w', encoding='utf-8') as f:
    f.write(content)
