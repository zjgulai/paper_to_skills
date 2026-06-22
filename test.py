import re

with open('playbook/assets/chat-page.js', 'r', encoding='utf-8') as f:
    text = f.read()

print("matchRiskEvent exists:", "matchRiskEvent" in text)
print("buildRAGContext exists:", "buildRAGContext" in text)
