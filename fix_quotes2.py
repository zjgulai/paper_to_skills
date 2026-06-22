with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(r"""let sys='你是 paper2skills 知识库的专业 AI 问答助手，专注于母婴跨境电商 AI 决策。\\n知识库现有 {skill_count} 个从顶会论文萃取的可落地业务技能。\\n回答规范：优先引用知识库中的具体 Skill，格式：[[Skill-具体名称]]；给出可操作具体建议。\\n当前时间：'+new Date().toLocaleDateString('zh-CN',{year:'numeric',month:'long',day:'numeric'});""", 
r"""let sys=`你是 paper2skills 知识库的专业 AI 问答助手，专注于母婴跨境电商 AI 决策。\n知识库现有 {skill_count} 个从顶会论文萃取的可落地业务技能。\n回答规范：优先引用知识库中的具体 Skill，格式：[[Skill-具体名称]]；给出可操作具体建议。\n当前时间：${new Date().toLocaleDateString('zh-CN',{year:'numeric',month:'long',day:'numeric'})}`;""")

with open('paper2skills-skills/playbook-generator/scripts/build_playbook.py', 'w', encoding='utf-8') as f:
    f.write(content)
