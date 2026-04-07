## Prompt 1: 核心场景 - 正常萃取流程

**场景**: 用户使用完整的萃取流程，生成一个uplift modeling的skill卡片

**输入**:
```
帮我将 arXiv:2103.02323 (Uplift Modeling) 萃取出 skill 卡片
```

**预期**:
- skill 被正确触发
- 执行完整流程：论文准备 → PDF下载 → Master Prompt → 代码生成 → 代码验证 → 保存
- 代码验证是强制流程，未通过不能保存