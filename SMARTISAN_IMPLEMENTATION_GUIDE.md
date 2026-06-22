# Paper2Skills Playbook - Smartisan 设计风格重设计实施指南

## 📋 项目目标

为 **paper2skills Playbook**（https://skills.lute-tlz-dddd.top/）应用 **Smartisan 设计语言**，实现：
- ✅ 克制感 + 科技感的视觉统一
- ✅ 高端化品牌感受
- ✅ 清晰的信息层级
- ✅ 移动友好的响应式设计

---

## 🎯 核心设计原则对标

### Smartisan vs 当前 Paper2Skills

| 维度 | 当前状态 | Smartisan 目标 | 实施方式 |
|------|---------|--------------|--------|
| 背景色 | 不统一 | 纯白 #fff + 微卡片背景 #f7f7f7 | 统一调色板 |
| 文字对比度 | 有落差 | 明确4级层级（主/次/三级/禁用） | 新增语义色彩变量 |
| 卡片设计 | 可能无边框 | 浅边框 #e0e0e0 + 微阴影 | 统一卡片组件库 |
| 按钮 | 风格混乱 | 深蓝实心 #0073e6 + 3状态（主/次/文字） | 3个按钮组件 |
| 圆角 | 不统一 | 统一 4-8px（克制感） | 圆角系统 |
| 间距 | 无栅格 | 8px 基础栅格 | 空间系统 |

---

## 🛠️ 实施路线图（5个阶段）

### Phase 1: 基础设施搭建（Day 1）
**目标**：引入 CSS 变量系统

```bash
# 1.1 创建设计 token 文件
touch paper2skills-skills/playbook-generator/src/styles/tokens.css

# 1.2 定义全局 CSS 变量（参考 smartisan_design_tokens.md 第七部分）
# 内容：色彩、排版、间距、圆角、阴影、动画
```

**检查清单**：
- [ ] `tokens.css` 包含完整 CSS 变量声明
- [ ] 在 `index.html` 或主样式表中引入 `tokens.css`
- [ ] 验证浏览器 DevTools 可读取变量值

---

### Phase 2: 核心色彩系统（Day 1-2）
**目标**：从当前色彩迁移到 Smartisan 调色板

```css
/* 关键替换 */
/* 旧 → 新 */
#fff → var(--color-white)           /* 保持 */
#000 → var(--text-primary)          /* #1a1a1a */
#666 → var(--text-secondary)        /* #2d2d2d */
#ccc → var(--border-light)          /* #e0e0e0 */
#f5f5f5 → var(--bg-secondary)       /* #f7f7f7 */
按钮蓝 → var(--accent-primary)      /* #0073e6（如果不同） */
```

**实施**：
1. 使用 `ast_grep_replace` 批量替换颜色值
2. 测试色彩对比度（WCAG AA 标准 ≥4.5:1）
3. 截图对比新旧色彩

---

### Phase 3: 排版统一（Day 2-3）
**目标**：实现 8 级排版阶梯

```css
/* 应用预设类 */
<h1 class="type-h1">大标题</h1>
<p class="type-body">正文</p>
<span class="type-caption">说明文本</span>
```

**实施**：
1. 在组件库中定义 `.type-*` 类
2. 更新所有标题/正文元素
3. 验证行高 & 字间距一致性

---

### Phase 4: 卡片 & 按钮标准化（Day 3-4）
**目标**：统一交互组件

```css
/* 卡片标准 */
.card {
  border: 1px solid var(--border-light);
  border-radius: var(--radius-md);
  padding: var(--space-6);
  box-shadow: var(--shadow-sm);
}

/* 主按钮 */
.btn-primary {
  background: var(--accent-primary);
  border-radius: var(--radius-md);
  padding: var(--space-3) var(--space-6);
}
```

**实施**：
1. 创建 `components/Card.tsx` & `components/Button.tsx`
2. 替换页面中所有硬编码样式
3. 测试 hover/active/disabled 状态

---

### Phase 5: 响应式 & 微交互（Day 4-5）
**目标**：完美的多端体验

```css
/* 响应式 */
@media (max-width: 768px) {
  --font-size-hero: 2rem;
  --space-10: 2rem;
}

/* 动画 */
.card {
  transition: var(--transition-default);
}
.card:hover {
  box-shadow: var(--shadow-md);
}
```

**实施**：
1. 应用响应式断点
2. 测试 hover/focus/active 状态
3. 跨浏览器验证

---

## 📁 文件修改清单

### 必改文件

```
paper2skills-skills/playbook-generator/
├── src/
│   ├── styles/
│   │   ├── tokens.css                  ← [新建] CSS 变量
│   │   ├── components.css              ← [修改] 卡片/按钮样式
│   │   └── global.css                  ← [修改] 全局色彩替换
│   ├── components/
│   │   ├── Card.tsx                    ← [新建或修改] 标准卡片
│   │   ├── Button.tsx                  ← [新建或修改] 3类按钮
│   │   ├── Input.tsx                   ← [新建或修改] 输入框
│   │   ├── Navbar.tsx                  ← [修改] 导航栏
│   │   └── Hero.tsx                    ← [修改] 首屏
│   ├── pages/
│   │   ├── Index.tsx                   ← [修改] 应用新色彩/排版
│   │   ├── Skills.tsx                  ← [修改]
│   │   └── Domains.tsx                 ← [修改]
│   └── App.tsx                         ← [修改] 引入 tokens.css
├── scripts/
│   └── build_playbook.py               ← [修改] 如需更新 HTML 生成
└── public/
    ├── index.html                      ← [修改] 链接 tokens.css
    └── favicon.ico                     ← [可选] 品牌优化
```

---

## 🔧 具体代码修改示例

### 1. 创建 `tokens.css`

```css
/* paper2skills-skills/playbook-generator/src/styles/tokens.css */

:root {
  /* ========== 颜色系统 ========== */
  --color-primary-900: #0066cc;
  --color-primary-700: #0073e6;
  /* ... 完整列表见 smartisan_design_tokens.md 第七部分 */
}
```

### 2. 修改全局样式

```css
/* before: global.css */
body {
  background: white;
  color: #333;
  font-family: -apple-system, BlinkMacSystemFont, sans-serif;
}

/* after: global.css */
body {
  background: var(--bg-primary);
  color: var(--text-primary);
  font-family: var(--font-family-base);
}

h1 {
  font-size: var(--font-size-h1);
  font-weight: var(--font-weight-h1);
  line-height: var(--line-height-h1);
}
```

### 3. 创建标准按钮组件

```tsx
// paper2skills-skills/playbook-generator/src/components/Button.tsx

export const Button = ({ 
  variant = 'primary',  // primary | secondary | text
  children, 
  ...props 
}) => {
  const className = {
    primary: 'btn-primary',
    secondary: 'btn-secondary',
    text: 'btn-text'
  }[variant];
  
  return <button className={className} {...props}>{children}</button>;
};
```

```css
/* button.css */
.btn-primary {
  padding: var(--space-3) var(--space-6);
  background: var(--accent-primary);
  color: var(--color-white);
  border-radius: var(--radius-md);
  border: none;
  transition: var(--transition-default);
}

.btn-primary:hover {
  background: var(--accent-hover);
  box-shadow: var(--shadow-md);
}
```

### 4. 批量替换颜色值

```bash
# 使用 ast_grep_replace 批量替换
# 示例：替换硬编码 #f5f5f5 为 var(--bg-secondary)

ast_grep_replace \
  --lang css \
  --pattern '#f5f5f5' \
  --rewrite 'var(--bg-secondary)' \
  --paths 'src/styles' 'src/components'
```

---

## ✅ 验收标准

### 色彩验收
- [ ] 所有背景使用 `var(--bg-*)` 系列
- [ ] 所有文本使用 `var(--text-*)` 系列
- [ ] 所有边框使用 `var(--border-*)` 系列
- [ ] WCAG AA 对比度通过（用 DevTools Lighthouse）

### 排版验收
- [ ] 所有标题使用 `class="type-h1/h2/h3"` 或 CSS 变量
- [ ] 行高 ≥1.4（正文）、≥1.2（标题）
- [ ] 字号跟随响应式断点

### 组件验收
- [ ] 所有卡片有 `border: 1px solid var(--border-light)`
- [ ] 所有卡片有 `box-shadow: var(--shadow-sm)`
- [ ] 所有按钮 hover 态有 `box-shadow: var(--shadow-md)`
- [ ] 所有圆角 ≤8px，大多数 6px

### 交互验收
- [ ] 所有过渡 150-300ms，使用 `var(--transition-*)`
- [ ] Focus 状态有清晰蓝色光晕（`box-shadow: 0 0 0 3px rgba(0, 115, 230, 0.1)`）
- [ ] 按钮 disabled 态有 `opacity: 0.6`

### 响应式验收
- [ ] 移动端 (<768px)：字号缩小、间距压缩
- [ ] 平板 (768-1024px)：中等缩放
- [ ] 桌面 (>1024px)：充分利用空间

---

## 🚀 快速启动命令

```bash
# 1. 复制设计 token
cp smartisan_design_tokens.md paper2skills-skills/playbook-generator/docs/

# 2. 创建 tokens.css（包含第七部分内容）
touch paper2skills-skills/playbook-generator/src/styles/tokens.css

# 3. 在主样式表中引入
echo "@import './tokens.css';" | \
  cat - paper2skills-skills/playbook-generator/src/styles/global.css \
  > /tmp/global.css && \
  mv /tmp/global.css paper2skills-skills/playbook-generator/src/styles/global.css

# 4. 运行 build（验证无错误）
cd paper2skills-skills/playbook-generator && npm run build

# 5. 本地预览
npm run dev
# 在 http://localhost:3000 检查色彩/排版/组件

# 6. 拍屏对比
# 新旧界面并排截图，对标 smartisan_design_tokens.md 的 token 表
```

---

## 📊 对标检查清单（Post-Implementation）

使用这个清单在完成后进行最终验收：

```markdown
### 色彩体系检查
- [ ] 主背景：纯白 #fff
- [ ] 次背景：#f7f7f7 或 #fafafa（卡片）
- [ ] 主文本：#1a1a1a（深黑）
- [ ] 次文本：#2d2d2d（深灰）
- [ ] 禁用文本：#666666（浅灰）
- [ ] 边框：#e0e0e0（极浅灰）
- [ ] 强调色：#0073e6（品牌蓝）

### 排版检查
- [ ] Hero 标题：56px, 700 weight, 1.1 line-height
- [ ] H1：40px, 600 weight, 1.2 line-height
- [ ] H2：32px, 600 weight, 1.25 line-height
- [ ] 正文：16px, 400 weight, 1.5 line-height
- [ ] 字体栈包含 -apple-system 和中文字体

### 组件检查
- [ ] 卡片：border 1px + border-light + radius-md + shadow-sm
- [ ] 主按钮：实心蓝 + radius-md + shadow-sm → shadow-md on hover
- [ ] 次按钮：透明底 + 蓝边框 + hover 背景色
- [ ] 输入框：radius-md + focus 蓝色光晕
- [ ] 导航栏：64px height + 白背景 + 下边框

### 交互检查
- [ ] 所有过渡 150-300ms
- [ ] Focus 有视觉反馈（蓝光晕或边框变色）
- [ ] Hover 有阴影增强或背景色变化
- [ ] Active 有压低感或颜色加深

### 响应式检查
- [ ] 320px（手机小屏）：可读、不溢出
- [ ] 768px（平板）：中等展示、两列可用
- [ ] 1280px（桌面）：充分空间、四列以上

### 无障碍检查
- [ ] 文字对比度 ≥4.5:1（正文）或 ≥3:1（标题）
- [ ] 交互元素 ≥44px（触摸目标）
- [ ] Focus 状态清晰可见
- [ ] 颜色不是唯一信息媒介（避免仅用颜色区分）
```

---

## 💡 常见问题 & 解决方案

### Q1: 如何快速验证 token 变量是否生效？
```bash
# 在浏览器 DevTools Console 中运行
getComputedStyle(document.documentElement).getPropertyValue('--color-primary-700')
# 应输出：#0073e6
```

### Q2: 如何确保颜色在所有浏览器中一致？
```css
/* 使用 RGB 备份 */
--color-primary-700: #0073e6;
--color-primary-700-rgb: 0, 115, 230;

/* 使用时 */
background: rgb(var(--color-primary-700-rgb) / 0.8); /* 带透明度 */
```

### Q3: 移动端字号如何自动缩放？
```css
/* 使用 clamp() 函数 */
--font-size-hero: clamp(2rem, 5vw, 3.5rem);
/* 最小 2rem，最大 3.5rem，根据视口宽度流体缩放 */
```

### Q4: 如何测试圆角是否统一？
```bash
# 使用这个 DevTools snippet
Array.from(document.querySelectorAll('*')).forEach(el => {
  const br = getComputedStyle(el).borderRadius;
  if (br && br !== '0px') console.log(el, br);
});
```

---

## 📚 参考资源

- **完整 Token 定义**：`smartisan_design_tokens.md`（同级目录）
- **Smartisan 官网**：https://www.smartisan.com/
- **CSS 变量参考**：https://developer.mozilla.org/en-US/docs/Web/CSS/--*
- **WCAG 无障碍**：https://www.w3.org/WAI/WCAG21/quickref/

---

## 🎬 Next Steps

1. **Day 1 Early**: 与设计团队确认 token 数值（尤其是品牌主色）
2. **Day 1 Mid**: 创建 `tokens.css` 并在主样式表中引入
3. **Day 2**: 批量替换颜色，验证 WCAG 对比度
4. **Day 3**: 统一卡片/按钮/输入框组件
5. **Day 4**: 测试响应式和微交互
6. **Day 5**: QA & 截图对标，准备上线

---

**Created**: 2026-06-18  
**Updated**: 2026-06-18  
**Version**: 1.0  
**Status**: Ready for Implementation ✅

