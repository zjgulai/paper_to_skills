with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

# Is there any other processing happening in `write_file`?
write_file_def = re.search(r'def write_file\(.*?\):.*?(\n\s*\n|\Z)', content, re.DOTALL)
if write_file_def:
    print(write_file_def.group(0))

