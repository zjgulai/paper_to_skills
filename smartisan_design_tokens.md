# Smartisan.com 设计语言深度分析 & 可移植 Token 库

## 一、核心设计哲学

### 1.1 Smartisan 品牌特征
- **克制感的来源**：极少装饰性元素、明确的信息层级、大量留白
- **科技感表达**：无衬线字体、直线元素、几何对齐、微妙渐变
- **品牌气质**：简约精致 + 高端感 + 易用性，对标 Apple 但更具东方审美

### 1.2 与对标的对比
| 维度 | Apple.com | Linear.app | Smartisan.com |
|------|-----------|-----------|---------------|
| 背景 | 纯白 #fff | 深灰 #050505 | 浅白/浅灰混搭 |
| 文字对比度 | 极高（黑+白） | 中等（灰+白） | 高（深灰+白） |
| 卡片样式 | 无卡片，网格布局 | 极简卡片，无边框 | 微卡片+边框 |
| 圆角 | 0-20px（产品图） | 8-12px（紧凑感） | 4-8px（克制感） |
| CTA 按钮 | 实心+圆角 | 描边+渐变 | 实心+细微阴影 |

---

## 二、设计 Token 库（可直接用于 CSS）

### 2.1 色彩体系

#### 主色板（Primary Palette）
```css
:root {
  /* 品牌主色 - 深蓝 */
  --color-primary-900: #0066cc;      /* 深蓝，最深，用于大面积品牌色 */
  --color-primary-700: #0073e6;      /* 中蓝，按钮/链接主状态 */
  --color-primary-500: #1a85ff;      /* 亮蓝，悬停/强调 */
  --color-primary-300: #4da6ff;      /* 浅蓝，禁用/次要状态 */
  
  /* 中性色系 - 最重要的色板 */
  --color-gray-950: #0a0a0a;         /* 纯黑，谨慎使用 */
  --color-gray-900: #1a1a1a;         /* 深黑，页面背景/主文本 */
  --color-gray-800: #2d2d2d;         /* 深灰，次级文本 */
  --color-gray-700: #4a4a4a;         /* 中灰，第三级文本 */
  --color-gray-600: #666666;         /* 灰，禁用/占位文本 */
  --color-gray-500: #999999;         /* 浅灰，边框/分割线 */
  --color-gray-400: #cccccc;         /* 更浅灰 */
  --color-gray-300: #e0e0e0;         /* 浅灰，浅边框 */
  --color-gray-200: #f0f0f0;         /* 浅灰，次级背景 */
  --color-gray-100: #f7f7f7;         /* 几乎白，卡片背景 */
  --color-gray-50: #fafafa;          /* 最浅，主背景/悬停 */
  --color-white: #ffffff;            /* 纯白 */
  
  /* 功能色 */
  --color-success: #22c55e;          /* 成功绿 */
  --color-warning: #f59e0b;          /* 警告橙 */
  --color-error: #ef4444;            /* 错误红 */
  --color-info: #06b6d4;             /* 信息青 */
}
```

#### 语义色彩映射
```css
/* 深色模式（主要模式） */
--bg-primary: var(--color-white);                    /* 页面主背景 */
--bg-secondary: var(--color-gray-50);               /* 次背景，卡片 */
--bg-tertiary: var(--color-gray-100);               /* 第三级背景，hover */
--bg-overlay: rgba(26, 26, 26, 0.5);               /* 模态遮罩 */

--text-primary: var(--color-gray-900);              /* 主文本，标题 */
--text-secondary: var(--color-gray-800);            /* 次文本，描述 */
--text-tertiary: var(--color-gray-700);             /* 第三级，辅助信息 */
--text-disabled: var(--color-gray-600);             /* 禁用文本 */
--text-inverse: var(--color-white);                 /* 反色文本（在深底上） */

--border-primary: var(--color-gray-500);            /* 主边框 */
--border-secondary: var(--color-gray-400);          /* 次边框，较淡 */
--border-light: var(--color-gray-300);              /* 浅边框，分割线 */

--accent-primary: var(--color-primary-700);         /* 强调色 */
--accent-hover: var(--color-primary-500);           /* 悬停强调 */
--accent-disabled: var(--color-primary-300);        /* 禁用强调 */
```

#### 色彩使用规则
| 场景 | 色值 | 说明 |
|------|------|------|
| 页面背景 | #ffffff | 纯白，极简 |
| 卡片背景 | #f7f7f7 或 #fafafa | 极浅灰，提供微层次 |
| 主标题 | #1a1a1a | 深黑，极高对比度 |
| 副标题 | #2d2d2d | 深灰，明显但不刺眼 |
| 描述文本 | #666666 | 中灰，可读但次要 |
| 禁用/占位 | #999999 | 浅灰，明确表示状态 |
| CTA 按钮 | #0066cc 到 #0073e6 | 品牌深蓝 |
| 边框/分割线 | #e0e0e0 | 浅灰，细致感 |

---

### 2.2 排版体系

#### 字体族
```css
/* 全局字体栈 */
--font-family-base: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, 
                     "Helvetica Neue", Arial, "PingFang SC", "HiragPro", 
                     "Microsoft YaHei UI", sans-serif;
--font-family-mono: "SF Mono", Monaco, "Cascadia Code", "Roboto Mono", Courier, monospace;

/* 字体说明：
   - -apple-system / BlinkMacSystemFont: macOS/iOS 原生
   - Segoe UI: Windows
   - Roboto: Android/Linux
   - PingFang SC / Microsoft YaHei UI: 中文优化
*/
```

#### 排版阶梯（8 级制）
```css
/* Display / Hero 级别 */
--font-size-hero: 3.5rem;           /* 56px, 特大标题（Hero/首屏） */
--line-height-hero: 1.1;            /* 紧凑 */
--font-weight-hero: 700;            /* 超粗 */
--letter-spacing-hero: -0.02em;     /* 紧凑 */

/* Heading 1 */
--font-size-h1: 2.5rem;             /* 40px, 页面标题 */
--line-height-h1: 1.2;
--font-weight-h1: 600;
--letter-spacing-h1: -0.01em;

/* Heading 2 */
--font-size-h2: 2rem;               /* 32px, 区域标题 */
--line-height-h2: 1.25;
--font-weight-h2: 600;
--letter-spacing-h2: 0;

/* Heading 3 */
--font-size-h3: 1.5rem;             /* 24px, 卡片标题 */
--line-height-h3: 1.33;
--font-weight-h3: 600;
--letter-spacing-h3: 0;

/* Heading 4 */
--font-size-h4: 1.25rem;            /* 20px, 小区块标题 */
--line-height-h4: 1.4;
--font-weight-h4: 500;
--letter-spacing-h4: 0;

/* Body / 正文 */
--font-size-body: 1rem;             /* 16px, 标准正文 */
--line-height-body: 1.5;            /* 24px line height，可读性最优 */
--font-weight-body: 400;            /* Normal */
--letter-spacing-body: 0;

/* Body Small */
--font-size-body-sm: 0.9375rem;     /* 15px, 次要正文 */
--line-height-body-sm: 1.5;
--font-weight-body-sm: 400;
--letter-spacing-body-sm: 0;

/* Caption / 说明 */
--font-size-caption: 0.875rem;      /* 14px, 图片说明、meta 信息 */
--line-height-caption: 1.43;
--font-weight-caption: 400;
--letter-spacing-caption: 0;

/* Overline / 超小标签 */
--font-size-overline: 0.75rem;      /* 12px, 标签、badge */
--line-height-overline: 1.33;
--font-weight-overline: 600;        /* 超粗以增加识别度 */
--letter-spacing-overline: 0.05em;  /* 字间距拉开，增加高级感 */
```

#### 排版组件类
```css
/* 预设文本样式类 */
.type-hero {
  font: var(--font-weight-hero) var(--font-size-hero) / var(--line-height-hero) var(--font-family-base);
  letter-spacing: var(--letter-spacing-hero);
  color: var(--text-primary);
}

.type-h1 {
  font: var(--font-weight-h1) var(--font-size-h1) / var(--line-height-h1) var(--font-family-base);
  letter-spacing: var(--letter-spacing-h1);
  color: var(--text-primary);
}

.type-h2 {
  font: var(--font-weight-h2) var(--font-size-h2) / var(--line-height-h2) var(--font-family-base);
  color: var(--text-primary);
}

.type-h3 {
  font: var(--font-weight-h3) var(--font-size-h3) / var(--line-height-h3) var(--font-family-base);
  color: var(--text-primary);
}

.type-body {
  font: var(--font-weight-body) var(--font-size-body) / var(--line-height-body) var(--font-family-base);
  color: var(--text-secondary);
}

.type-caption {
  font: var(--font-weight-caption) var(--font-size-caption) / var(--line-height-caption) var(--font-family-base);
  color: var(--text-tertiary);
}

.type-label {
  font: var(--font-weight-overline) var(--font-size-overline) / var(--line-height-overline) var(--font-family-base);
  letter-spacing: var(--letter-spacing-overline);
  text-transform: uppercase;
  color: var(--text-secondary);
}
```

---

### 2.3 间距体系（8px 基础栅格）

```css
/* 间距阶梯 - 8px 为基础单位 */
--space-0: 0;
--space-1: 0.25rem;      /* 4px */
--space-2: 0.5rem;       /* 8px - 基础单位 */
--space-3: 0.75rem;      /* 12px */
--space-4: 1rem;         /* 16px - 标准间距 */
--space-5: 1.25rem;      /* 20px */
--space-6: 1.5rem;       /* 24px - 常用 */
--space-7: 2rem;         /* 32px - 大段落间距 */
--space-8: 2.5rem;       /* 40px */
--space-9: 3rem;         /* 48px - 大间隔 */
--space-10: 4rem;        /* 64px - 极大间隔 */

/* 快速引用 */
--gap-xs: var(--space-2);    /* 8px */
--gap-sm: var(--space-3);    /* 12px */
--gap-md: var(--space-4);    /* 16px - 默认 */
--gap-lg: var(--space-6);    /* 24px */
--gap-xl: var(--space-9);    /* 48px */

/* 内边距规则 */
--padding-xs: var(--space-2);       /* 8px */
--padding-sm: var(--space-3);       /* 12px */
--padding-md: var(--space-4);       /* 16px */
--padding-lg: var(--space-6);       /* 24px */
--padding-xl: var(--space-8);       /* 40px */

/* 页面级外边距 */
--margin-page-top: var(--space-9);     /* 48px */
--margin-page-side: 5%;                 /* 响应式侧边距 */

/* 卡片间距 */
--card-padding: var(--space-4);        /* 内边距 16px */
--card-gap: var(--space-4);            /* 卡片间间距 */
```

---

### 2.4 圆角系统

```css
/* 圆角半径 - 体现 Smartisan 的克制感 */
--radius-none: 0;
--radius-sm: 4px;          /* 最小，用于细微UI元素 */
--radius-md: 6px;          /* 标准，卡片/按钮 */
--radius-lg: 8px;          /* 较大，模态/输入框 */
--radius-full: 9999px;     /* 无限大，用于圆形/胶囊 */

/* 分类使用 */
--border-radius-button: var(--radius-md);      /* 按钮 6px */
--border-radius-input: var(--radius-md);       /* 输入框 6px */
--border-radius-card: var(--radius-md);        /* 卡片 6px */
--border-radius-modal: var(--radius-lg);       /* 模态 8px */
--border-radius-avatar: var(--radius-full);    /* 头像 圆形 */
```

---

### 2.5 阴影系统（微妙、专业）

```css
/* 阴影 - Smartisan 采用极微妙的投影，增加分层感而不显突兀 */

/* 无阴影（默认） */
--shadow-none: none;

/* 微阴影 - 用于卡片、按钮轻提升 */
--shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05),
             0 1px 3px rgba(0, 0, 0, 0.1);

/* 标准阴影 - 用于卡片、下拉菜单 */
--shadow-md: 0 4px 6px rgba(0, 0, 0, 0.07),
             0 10px 13px rgba(0, 0, 0, 0.1);

/* 大阴影 - 用于模态、弹出层 */
--shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.1),
             0 25px 25px rgba(0, 0, 0, 0.15);

/* 极大阴影 - 罕用，特殊强调 */
--shadow-xl: 0 20px 25px rgba(0, 0, 0, 0.1),
             0 10px 10px rgba(0, 0, 0, 0.2);

/* 内阴影 - 凹陷感，输入框/深色背景 */
--shadow-inset: inset 0 2px 4px rgba(0, 0, 0, 0.06);

/* 应用场景 */
--shadow-card: var(--shadow-sm);               /* 卡片 */
--shadow-button-hover: var(--shadow-md);       /* 按钮悬停 */
--shadow-dropdown: var(--shadow-md);           /* 下拉菜单 */
--shadow-modal: var(--shadow-lg);              /* 模态弹窗 */
```

---

### 2.6 边框系统

```css
/* 边框宽度 */
--border-width-thin: 1px;      /* 标准边框 */
--border-width-medium: 2px;    /* 强调边框 */
--border-width-thick: 3px;     /* 活跃/选中 */

/* 边框样式 */
--border-style-solid: solid;
--border-style-dashed: dashed;

/* 组合应用 */
--border-default: var(--border-width-thin) var(--border-style-solid) var(--border-primary);
--border-light: var(--border-width-thin) var(--border-style-solid) var(--border-light);
--border-focus: var(--border-width-thin) var(--border-style-solid) var(--accent-primary);

/* 分割线 */
--divider-horizontal: 1px solid var(--border-light);
--divider-vertical: 1px solid var(--border-light);
```

---

### 2.7 动画 & 过渡

```css
/* 过渡时间 - 微妙但清晰 */
--transition-fast: 150ms;       /* 快速反馈（按钮、悬停） */
--transition-normal: 300ms;     /* 标准过渡（淡入/淡出） */
--transition-slow: 500ms;       /* 缓慢动画（页面进入） */

/* 缓动函数 - 保持专业感 */
--timing-ease-out: cubic-bezier(0.4, 0, 0.2, 1);      /* 快出 */
--timing-ease-in-out: cubic-bezier(0.4, 0, 0.2, 1);   /* 均匀 */
--timing-ease-in: cubic-bezier(0.4, 0, 1, 1);         /* 缓进 */

/* 预设过渡 */
--transition-default: all var(--transition-normal) var(--timing-ease-in-out);
--transition-color: color var(--transition-fast) var(--timing-ease-in-out);
--transition-shadow: box-shadow var(--transition-fast) var(--timing-ease-in-out);
```

---

## 三、UI 组件设计规范

### 3.1 按钮设计

#### 主按钮（Primary CTA）
```css
/* 实心蓝色，圆角，微阴影 */
.btn-primary {
  padding: var(--space-3) var(--space-6);           /* 12px 24px */
  background-color: var(--accent-primary);         /* #0073e6 */
  color: var(--color-white);
  border: none;
  border-radius: var(--radius-md);                 /* 6px */
  font-size: var(--font-size-body);
  font-weight: 500;
  cursor: pointer;
  transition: var(--transition-default);
  box-shadow: var(--shadow-sm);
}

.btn-primary:hover {
  background-color: var(--accent-hover);           /* #1a85ff */
  box-shadow: var(--shadow-md);
}

.btn-primary:active {
  background-color: var(--color-primary-900);      /* #0066cc */
}

.btn-primary:disabled {
  background-color: var(--accent-disabled);        /* #4da6ff */
  cursor: not-allowed;
  opacity: 0.6;
}
```

#### 次按钮（Secondary）
```css
/* 描边，无背景，极简 */
.btn-secondary {
  padding: var(--space-3) var(--space-6);
  background-color: transparent;
  color: var(--accent-primary);
  border: 1px solid var(--accent-primary);
  border-radius: var(--radius-md);
  font-size: var(--font-size-body);
  font-weight: 500;
  cursor: pointer;
  transition: var(--transition-default);
}

.btn-secondary:hover {
  background-color: var(--bg-tertiary);            /* #f0f0f0 */
  color: var(--accent-hover);
  border-color: var(--accent-hover);
}
```

#### 文字按钮（Tertiary）
```css
/* 纯文字，无边框/背景 */
.btn-text {
  padding: var(--space-2) var(--space-3);
  background: none;
  border: none;
  color: var(--accent-primary);
  font-size: var(--font-size-body);
  font-weight: 500;
  cursor: pointer;
  transition: var(--transition-color);
  text-decoration: none;
}

.btn-text:hover {
  color: var(--accent-hover);
  text-decoration: underline;
}
```

### 3.2 卡片设计

```css
/* 标准卡片 - Smartisan 的卡片设计极简 */
.card {
  background-color: var(--bg-secondary);           /* #f7f7f7 */
  border: 1px solid var(--border-light);           /* #e0e0e0 */
  border-radius: var(--radius-md);                 /* 6px */
  padding: var(--space-6);                         /* 24px */
  box-shadow: var(--shadow-sm);
  transition: var(--transition-default);
}

.card:hover {
  box-shadow: var(--shadow-md);
  border-color: var(--border-primary);             /* 边框变深 */
}

.card--interactive {
  cursor: pointer;
}

.card--elevated {
  background-color: var(--bg-primary);             /* 更亮 */
  box-shadow: var(--shadow-md);
}

/* 卡片内部结构 */
.card__header {
  padding-bottom: var(--space-4);
  border-bottom: var(--divider-horizontal);
}

.card__title {
  margin: 0;
  font-size: var(--font-size-h3);
  font-weight: 600;
  color: var(--text-primary);
}

.card__body {
  padding: var(--space-4) 0;
}

.card__footer {
  padding-top: var(--space-4);
  border-top: var(--divider-horizontal);
  display: flex;
  gap: var(--gap-md);
  justify-content: flex-end;
}
```

### 3.3 输入框设计

```css
.input {
  padding: var(--space-3) var(--space-4);          /* 12px 16px */
  background-color: var(--bg-primary);             /* 白底 */
  border: 1px solid var(--border-light);           /* #e0e0e0 */
  border-radius: var(--radius-md);                 /* 6px */
  font-size: var(--font-size-body);
  font-family: var(--font-family-base);
  color: var(--text-primary);
  transition: var(--transition-default);
}

.input:focus {
  outline: none;
  border-color: var(--accent-primary);
  box-shadow: 0 0 0 3px rgba(0, 115, 230, 0.1);   /* 蓝色光晕 */
}

.input:disabled {
  background-color: var(--bg-tertiary);
  color: var(--text-disabled);
  cursor: not-allowed;
}

.input::placeholder {
  color: var(--text-tertiary);
}
```

### 3.4 导航栏设计

```css
.navbar {
  position: sticky;
  top: 0;
  height: 64px;                                    /* 标准导航高度 */
  background-color: var(--bg-primary);             /* 纯白背景 */
  border-bottom: 1px solid var(--border-light);    /* #e0e0e0 */
  display: flex;
  align-items: center;
  padding: 0 var(--space-6);                       /* 24px 侧边距 */
  gap: var(--gap-lg);
  z-index: 100;
  box-shadow: var(--shadow-sm);
}

.navbar__logo {
  height: 32px;
  width: auto;
  margin-right: auto;
}

.navbar__menu {
  display: flex;
  gap: var(--gap-lg);
  list-style: none;
  margin: 0;
  padding: 0;
}

.navbar__item {
  color: var(--text-secondary);
  font-size: var(--font-size-body);
  font-weight: 500;
  cursor: pointer;
  transition: var(--transition-color);
  padding: var(--space-2) var(--space-3);
}

.navbar__item:hover,
.navbar__item--active {
  color: var(--text-primary);
  border-bottom: 2px solid var(--accent-primary);
}
```

---

## 四、Hero & 首屏设计规范

### 4.1 Hero 区块
```css
.hero {
  padding: var(--space-10) var(--space-6);         /* 顶部 64px，侧边 24px */
  background: linear-gradient(135deg, 
              var(--bg-primary) 0%, 
              var(--bg-secondary) 100%);           /* 微妙渐变 */
  text-align: center;
}

.hero__title {
  font-size: var(--font-size-hero);                /* 56px */
  font-weight: 700;
  line-height: var(--line-height-hero);
  letter-spacing: -0.02em;
  color: var(--text-primary);
  margin: 0 0 var(--space-4) 0;
  max-width: 900px;
  margin-left: auto;
  margin-right: auto;
}

.hero__subtitle {
  font-size: var(--font-size-h2);                  /* 32px */
  font-weight: 400;
  line-height: 1.4;
  color: var(--text-secondary);
  max-width: 800px;
  margin: 0 auto var(--space-6) auto;
}

.hero__cta-group {
  display: flex;
  gap: var(--gap-md);
  justify-content: center;
  flex-wrap: wrap;
}

.hero__cta {
  padding: var(--space-4) var(--space-8);          /* 16px 40px */
  font-size: var(--font-size-body);
  font-weight: 600;
}
```

---

## 五、响应式断点

```css
/* 移动优先设计 */
--breakpoint-sm: 640px;        /* 平板竖屏 */
--breakpoint-md: 768px;        /* 平板横屏 */
--breakpoint-lg: 1024px;       /* 桌面小屏 */
--breakpoint-xl: 1280px;       /* 桌面标准 */
--breakpoint-2xl: 1536px;      /* 超宽屏 */

/* 媒体查询示例 */
@media (max-width: 768px) {
  --font-size-hero: 2rem;      /* 移动端缩小 */
  --padding-page: var(--space-4);
}

@media (min-width: 1280px) {
  --padding-page: var(--space-8);
}
```

---

## 六、对标对比总结

### Smartisan 相比 Apple.com 的特色
| 方面 | Apple | Smartisan | 原因 |
|------|-------|----------|------|
| 背景 | 纯白 #fff | 浅白/灰混 | 东方审美，温暖感 |
| 圆角 | 0-20px 产品 | 4-8px 统一 | 克制感，高端感 |
| 阴影 | 几乎无 | 微妙投影 | 增加分层而保持简约 |
| 文字对比 | 极端黑白 | 中等对比 | 易读但不刺眼 |
| 按钮 | 纯描边 | 实心+微影 | 更明确的可操作性 |
| 间距 | 大量留白 | 适度留白 | 信息密度平衡 |

### Smartisan 相比 Linear.app 的特色
| 方面 | Linear | Smartisan | 原因 |
|------|--------|----------|------|
| 亮度 | 深灰为主 | 浅白为主 | 中文市场习惯，更明快 |
| 圆角 | 8-12px | 4-6px | 线性app 偏圆润，Smartisan 偏直 |
| 对比度 | 低对比 | 中-高对比 | Smartisan 更注重清晰度 |
| 卡片 | 无边框 | 有浅边框 | 增加分层感，亚洲审美 |
| CTA | 渐变+特殊 | 实心+经典 | Smartisan 更保守/成熟 |

---

## 七、CSS 变量声明完整示例

```css
:root {
  /* ========== 颜色 ========== */
  --color-primary-900: #0066cc;
  --color-primary-700: #0073e6;
  --color-primary-500: #1a85ff;
  --color-primary-300: #4da6ff;
  
  --color-gray-950: #0a0a0a;
  --color-gray-900: #1a1a1a;
  --color-gray-800: #2d2d2d;
  --color-gray-700: #4a4a4a;
  --color-gray-600: #666666;
  --color-gray-500: #999999;
  --color-gray-400: #cccccc;
  --color-gray-300: #e0e0e0;
  --color-gray-200: #f0f0f0;
  --color-gray-100: #f7f7f7;
  --color-gray-50: #fafafa;
  --color-white: #ffffff;
  
  --color-success: #22c55e;
  --color-warning: #f59e0b;
  --color-error: #ef4444;
  --color-info: #06b6d4;
  
  /* 语义色彩 */
  --bg-primary: var(--color-white);
  --bg-secondary: var(--color-gray-50);
  --bg-tertiary: var(--color-gray-100);
  --text-primary: var(--color-gray-900);
  --text-secondary: var(--color-gray-800);
  --text-tertiary: var(--color-gray-700);
  --text-disabled: var(--color-gray-600);
  --border-primary: var(--color-gray-500);
  --border-secondary: var(--color-gray-400);
  --border-light: var(--color-gray-300);
  --accent-primary: var(--color-primary-700);
  --accent-hover: var(--color-primary-500);
  --accent-disabled: var(--color-primary-300);
  
  /* ========== 排版 ========== */
  --font-family-base: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, 
                      "Helvetica Neue", Arial, "PingFang SC", "Microsoft YaHei UI", sans-serif;
  --font-family-mono: "SF Mono", Monaco, "Cascadia Code", monospace;
  
  --font-size-hero: 3.5rem;
  --font-size-h1: 2.5rem;
  --font-size-h2: 2rem;
  --font-size-h3: 1.5rem;
  --font-size-h4: 1.25rem;
  --font-size-body: 1rem;
  --font-size-body-sm: 0.9375rem;
  --font-size-caption: 0.875rem;
  --font-size-overline: 0.75rem;
  
  /* ========== 间距 ========== */
  --space-0: 0;
  --space-1: 0.25rem;
  --space-2: 0.5rem;
  --space-3: 0.75rem;
  --space-4: 1rem;
  --space-5: 1.25rem;
  --space-6: 1.5rem;
  --space-7: 2rem;
  --space-8: 2.5rem;
  --space-9: 3rem;
  --space-10: 4rem;
  
  --gap-xs: var(--space-2);
  --gap-sm: var(--space-3);
  --gap-md: var(--space-4);
  --gap-lg: var(--space-6);
  --gap-xl: var(--space-9);
  
  /* ========== 圆角 ========== */
  --radius-sm: 4px;
  --radius-md: 6px;
  --radius-lg: 8px;
  --radius-full: 9999px;
  
  /* ========== 阴影 ========== */
  --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1);
  --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.07), 0 10px 13px rgba(0, 0, 0, 0.1);
  --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.1), 0 25px 25px rgba(0, 0, 0, 0.15);
  
  /* ========== 动画 ========== */
  --transition-fast: 150ms;
  --transition-normal: 300ms;
  --transition-slow: 500ms;
  --timing-ease-out: cubic-bezier(0.4, 0, 0.2, 1);
  --timing-ease-in-out: cubic-bezier(0.4, 0, 0.2, 1);
  
  /* ========== 响应式 ========== */
  --breakpoint-sm: 640px;
  --breakpoint-md: 768px;
  --breakpoint-lg: 1024px;
  --breakpoint-xl: 1280px;
}
```

