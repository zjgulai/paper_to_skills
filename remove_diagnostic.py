with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'w', encoding='utf-8') as f:
    for line in lines:
        if "render_diagnostic_page" not in line:
            f.write(line)

