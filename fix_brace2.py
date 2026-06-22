with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace("    }}}\n    .cmsg-typing .cmsg-bubble::after {{", "    }}\n    .cmsg-typing .cmsg-bubble::after {{")

with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'w', encoding='utf-8') as f:
    f.write(content)
