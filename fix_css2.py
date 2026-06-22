with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Add to build_css directly
css_addon = """
/* Chat Badges */
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
"""

# Let's insert it before the closing of build_css string
if ".cmsg-event-badge" not in content.split("def build_css() -> str:")[1]:
    idx = content.find('"""\n\n\n# ---------------------------------------------------------------------------', content.find("def build_css() -> str:"))
    if idx != -1:
        content = content[:idx] + css_addon + content[idx:]
        with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'w', encoding='utf-8') as f:
            f.write(content)
        print("CSS added to build_css")
    else:
        print("Could not find end of build_css")
else:
    print("Already in build_css")
