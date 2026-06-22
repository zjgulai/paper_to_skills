with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

# Output the context of the second searchSkills
idx = 358462
print("Context around second searchSkills:")
print(content[idx-500:idx+500])
