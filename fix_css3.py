with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Ah! It's in `render_chat_page`'s inline CSS, not `build_css`! Let's check `build_css`.
css_idx = content.find("def build_css() -> str:")
end_css_idx = content.find('"""\n\n\n# ---------------------------------------------------------------------------', css_idx)

if ".cmsg-event-badge" not in content[css_idx:end_css_idx]:
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
    content = content[:end_css_idx] + css_addon + content[end_css_idx:]
    with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Added to build_css!")

