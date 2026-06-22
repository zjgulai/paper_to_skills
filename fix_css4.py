with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Let's inspect where `build_css` ends.
start_idx = content.find("def build_css() -> str:")
end_idx = content.find('"""\n\n\n# ---------------------------------------------------------------------------\n# Index page', start_idx)

if start_idx != -1 and end_idx != -1:
    css_addon = """
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
    content = content[:end_idx] + css_addon + content[end_idx:]
    with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Added!")
