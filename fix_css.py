import re

with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'r', encoding='utf-8') as f:
    content = f.read()

# I also need to add the CSS to `build_css()`
css_addon = """
.cmsg-event-badge {
  display: inline-flex; align-items: center; gap: 4px;
  font-size: 11px; color: #991b1b; margin-bottom: 6px;
  padding: 2px 8px; background: #fef2f2; border-radius: var(--r-full);
  border: 1px solid #fecaca;
}
"""

if ".cmsg-event-badge {" not in content.split("def build_css() -> str:")[1]:
    # Let's insert it after `.cmsg-web-badge { ... }`
    replacement = """.cmsg-web-badge {
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
}"""
    
    content = re.sub(r'\.cmsg-web-badge \{\n  display: inline-flex;.*?border: 1px solid var\(--line\);\n\}', replacement, content, flags=re.DOTALL)
    
    with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Added CSS to build_css()")
else:
    print("CSS already in build_css()")

