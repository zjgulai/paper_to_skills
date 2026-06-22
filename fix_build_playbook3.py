with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the CSS escaping in the f-string correctly
content = content.replace("  5832\t    }", "  5832\t    }}") # this might not match exactly due to tabs/spaces

# More reliable replace
import re
content = re.sub(r'    \.cmsg-web-badge \{\{.*?border: 1px solid var\(--line\);\n    \}', 
                 r"""    .cmsg-web-badge {{
      display: inline-flex; align-items: center; gap: 4px;
      font-size: 11px; color: var(--muted); margin-bottom: 6px;
      padding: 2px 8px; background: var(--panel-2); border-radius: var(--r-full);
      border: 1px solid var(--line);
    }}""", content, flags=re.DOTALL)

with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'w', encoding='utf-8') as f:
    f.write(content)
