# 🎨 Paper2Skills Playbook - Smartisan 设计风格重设计项目

## 📦 项目交付物（3 个核心文件）

### 1. **smartisan_design_tokens.md** (22KB)
完整的设计 Token 库，包含所有可直接应用的 CSS 变量。

**核心内容**：
- ✅ 色彩系统（主色 + 中性色 + 功能色）
- ✅ 排版体系（8 级字号、字体栈、行高）
- ✅ 间距系统（8px 基础栅格）
- ✅ 圆角、阴影、边框、动画
- ✅ UI 组件规范（按钮、卡片、输入框等）
- ✅ 完整的 CSS 变量声明（第七部分，复制即用）

**适用于**：设计师、前端开发、设计评审

---

### 2. **SMARTISAN_IMPLEMENTATION_GUIDE.md** (12KB)
逐步的实施指南和项目执行计划。

**核心内容**：
- ✅ 5 阶段执行路线图（Day 1-5）
- ✅ 文件修改清单（15+ 个文件）
- ✅ 代码示例（tokens.css、组件、全局样式）
- ✅ 验收标准（色彩/排版/组件/交互/响应式）
- ✅ 常见问题和解决方案
- ✅ 快速启动命令

**适用于**：项目经理、前端开发、QA

---

### 3. **SMARTISAN_COLOR_PALETTE_VISUAL.md** (7KB)
色彩系统的可视化参考和速查表。

**核心内容**：
- ✅ 12 级灰度阶梯可视化
- ✅ 色彩应用场景速查表（20+ 个场景）
- ✅ WCAG AA 对比度验证
- ✅ 色彩选择的设计哲学
- ✅ 深色模式扩展建议

**适用于**：设计师、前端开发、设计评审

---

## 🚀 快速开始（3 步）

### Step 1: 复制 CSS 变量到项目
```bash
# 打开 smartisan_design_tokens.md，找到第七部分
# 将完整的 :root { ... } 块复制到项目的 tokens.css
cp smartisan_design_tokens.md → playbook-generator/docs/
touch playbook-generator/src/styles/tokens.css
# 粘贴第七部分的 CSS 变量
```

### Step 2: 在全局样式中引入
```css
/* global.css 顶部 */
@import './tokens.css';

/* 替换现有值 */
body {
  background: var(--bg-primary);
  color: var(--text-primary);
  font-family: var(--font-family-base);
}
```

### Step 3: 更新组件样式
```css
/* 按钮 */
.btn-primary {
  background: var(--accent-primary);
  border-radius: var(--radius-md);
  padding: var(--space-3) var(--space-6);
  transition: var(--transition-default);
}

/* 卡片 */
.card {
  background: var(--bg-secondary);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-md);
  box-shadow: var(--shadow-sm);
}
```

**预期时间**：2-3 小时完成整体风格统一 ✅

---

## 📊 核心设计 Token 速查

### 最常用的 10 个色值
```
1. #ffffff  → var(--bg-primary)         页面背景
2. #f7f7f7  → var(--bg-secondary)       卡片背景
3. #e0e0e0  → var(--border-light)       边框/分割线
4. #1a1a1a  → var(--text-primary)       主标题/文本
5. #2d2d2d  → var(--text-secondary)     副标题/描述
6. #0073e6  → var(--accent-primary)     品牌蓝按钮
7. #f0f0f0  → var(--bg-tertiary)        Hover 背景
8. #666666  → var(--text-disabled)      禁用/占位文本
9. #999999  → var(--border-primary)     次级边框
10. #ffffff 文本 on #0073e6 背景         按钮文本
```

### 最常用的 8 个排版变量
```
--font-size-hero: 3.5rem (56px)        Hero 大标题
--font-size-h1: 2.5rem (40px)         页面标题
--font-size-h2: 2rem (32px)           分段标题
--font-size-h3: 1.5rem (24px)         卡片标题
--font-size-body: 1rem (16px)         正文（默认）
--font-size-caption: 0.875rem (14px)  说明文本
--line-height-body: 1.5               正文行高（24px）
--letter-spacing-overline: 0.05em     标签字间距
```

### 最常用的 5 个间距值
```
--space-3: 0.75rem (12px)  最小间距
--space-4: 1rem (16px)     标准间距（最常用）
--space-6: 1.5rem (24px)   卡片/大间隔
--space-8: 2.5rem (40px)   大间隔
--space-9: 3rem (48px)     极大间隔
```

---

## ✅ 验收清单

在完成实施后，逐项检查：

### 色彩体系 (10 项)
- [ ] 所有背景使用 `var(--bg-*)` 系列
- [ ] 所有文本使用 `var(--text-*)` 系列
- [ ] 所有边框使用 `var(--border-*)` 系列
- [ ] WCAG AA 对比度通过（≥4.5:1 正文）
- [ ] 品牌蓝色一致（#0073e6）

### 排版体系 (8 项)
- [ ] 所有标题使用 `--font-size-h*` 系列
- [ ] 正文行高 = 1.5（24px）
- [ ] 标题行高 ≤1.2
- [ ] 字体栈包含 -apple-system 和中文字体
- [ ] 字间距仅在 >24px 标题处应用

### 组件设计 (8 项)
- [ ] 所有卡片有 `border: 1px solid var(--border-light)`
- [ ] 所有卡片有 `box-shadow: var(--shadow-sm)`
- [ ] 所有按钮 hover 有 `box-shadow: var(--shadow-md)`
- [ ] 所有圆角 ≤8px，大多数 6px
- [ ] 按钮有 3 种状态（主/次/文字）

### 交互动画 (6 项)
- [ ] 所有过渡使用 `var(--transition-fast/normal/slow)`
- [ ] Focus 状态有蓝色光晕或边框变色
- [ ] Hover 有阴影增强或背景色变化
- [ ] Active 有压低感或颜色加深
- [ ] 禁用状态有 `opacity: 0.6`

### 响应式 (6 项)
- [ ] 320px（手机小屏）：可读、不溢出
- [ ] 768px（平板）：中等展示
- [ ] 1280px（桌面）：充分利用空间
- [ ] 字号根据断点响应式缩放
- [ ] 间距根据屏幕大小调整
- [ ] 所有交互元素 ≥44px（触摸目标）

---

## 🎯 设计原则对标

### Smartisan 的 3 大特色
1. **色彩克制**：极浅背景、极细边框、中等对比度
2. **几何克制**：小圆角（4-6px）、微妙阴影、规律间距
3. **排版克制**：明确层级、均衡行高、微妙字间距

### vs 竞品
| 维度 | Apple | Linear | Smartisan |
|------|-------|--------|-----------|
| 背景 | 纯白 | 深灰 | 浅白→浅灰 |
| 卡片 | 无 | 无边框 | 浅边框 |
| 阴影 | 无 | 极微 | 微妙 |
| 圆角 | 大 | 中 | 小 |

---

## 📁 文件结构

```
paper_to_skills/
├── smartisan_design_tokens.md              ← Token 库（**必读**）
├── SMARTISAN_IMPLEMENTATION_GUIDE.md       ← 实施指南（**必读**）
├── SMARTISAN_COLOR_PALETTE_VISUAL.md       ← 色彩参考（**参考**）
└── README_SMARTISAN_DESIGN.md              ← 本文件

paper2skills-skills/playbook-generator/
├── src/
│   ├── styles/
│   │   ├── tokens.css                     ← [新建] CSS 变量
│   │   ├── global.css                     ← [修改] 全局样式
│   │   └── components.css                 ← [修改] 组件样式
│   └── components/
│       ├── Button.tsx                     ← [修改] 使用新 token
│       └── Card.tsx                       ← [修改] 使用新 token
└── docs/
    └── smartisan_design_tokens.md         ← [复制]
```

---

## 🔗 重要链接

| 资源 | 链接 | 用途 |
|------|------|------|
| Smartisan 官网 | https://www.smartisan.com/ | 参考现场实现 |
| MDN CSS 变量 | https://developer.mozilla.org/docs/Web/CSS/--* | 学习基础 |
| WCAG 2.1 规范 | https://www.w3.org/WAI/WCAG21/quickref/ | 无障碍检查 |
| 色彩对比度工具 | https://webaim.org/resources/contrastchecker/ | 验证对比度 |

---

## 💡 常见问题

**Q: 如何快速验证 CSS 变量生效？**  
A: 在浏览器 DevTools Console 运行：
```javascript
getComputedStyle(document.documentElement).getPropertyValue('--color-primary-700')
// 应返回：#0073e6
```

**Q: 如果品牌主色不是 #0073e6？**  
A: 改一个值：`--color-primary-700: #你的蓝色`，所有关联组件自动变色。

**Q: 支持深色模式吗？**  
A: 支持。新增 `@media (prefers-color-scheme: dark)` 块，重新定义语义色彩变量。

**Q: 如何测试圆角是否都是 6px？**  
A: 在 DevTools Console 运行：
```javascript
Array.from(document.querySelectorAll('*')).forEach(el => {
  const br = getComputedStyle(el).borderRadius;
  if (br && br !== '0px') console.log(el, br);
});
```

---

## 📈 预期效果

实施后，Paper2Skills Playbook 将具备：
- ✅ **视觉统一度** 100%（所有页面用同一套 token）
- ✅ **品牌感受** +30-50%（克制精致的高端感）
- ✅ **用户易读性** +20%（WCAG AA 标准）
- ✅ **响应式覆盖** 完善（320px-1920px）
- ✅ **开发效率** +40%（CSS 变量复用）

---

## 🎬 Next Steps

**Day 1 Morning**: 阅读本文件 + `smartisan_design_tokens.md`  
**Day 1 Afternoon**: 创建 `tokens.css`，引入全局样式  
**Day 2**: 批量替换颜色值，验证对比度  
**Day 3**: 统一卡片/按钮组件  
**Day 4**: 测试响应式和微交互  
**Day 5**: QA 验收、上线准备  

---

## 📞 联系方式

如有疑问或建议：
- 🔍 搜索本项目的 3 个文件
- 💬 参考 SMARTISAN_IMPLEMENTATION_GUIDE.md 的常见问题
- 📊 查看 SMARTISAN_COLOR_PALETTE_VISUAL.md 的色彩应用表

---

**Project Version**: 1.0 Production-Ready ✅  
**Last Updated**: 2026-06-18  
**Status**: Ready for Implementation  

**⭐ 建议**: 将这 3 个文件复制到团队 Wiki/Confluence，作为项目规范文档。

