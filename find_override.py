with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

# Is there any other function with `_built=false`? Let's check where the minified JS is coming from.
matches = [m.start() for m in re.finditer(r'function searchSkills', content)]
print("Indices of searchSkills:", matches)
