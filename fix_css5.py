with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'r', encoding='utf-8') as f:
    content = f.read()

import re

# Add CSS right after body{
content = re.sub(r'body\s*\{', r'''
.cmsg-web-badge {
  display: inline-flex; align-items: center; gap: 4px;
  font-size: 11px; color: var(--muted); margin-bottom: 6px;
  padding: 2px 8px; background: var(--panel-2); border-radius: var(--r-full);
  border: 1px solid var(--line);
}
.cmsg-event-badge {
  display: inline-flex; align-items: center; gap: 4px;
  font-size: 11px; color: #991b1b; margin-bottom: 6px;
  padding: 2px 8px; background: #fef2f2; border-radius: var(--r-full);
  border: 1px solid #fecaca;
}
body {''', content, count=1)

with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Done")
