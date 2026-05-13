# NLP-VOC Presentation

单页滚动 HTML 项目方案与进度汇报。

## 当前阶段

| 阶段 | 状态 | 内容 |
|---|---|---|
| Step 1 | ✅ 已通过 | 2 段 showcase 风格审阅 |
| Step 2 | ✅ 已通过 | 主线 16 段 + 附录 6 段骨架 |
| Step 3 | ✅ 已通过 | 内容终稿 + 5 ECharts + 3 SVG + 4 dashboard mockup + 离线 vendor |
| Step 4 | ✅ 已通过 | PDF 导出（主版 22 页 + with-notes 44 页），0 page error |

## 目录结构

```
presentation/
├── index.html                              单页 22 段汇报
├── README.md
├── showcase/                               Step 1 风格审阅页（保留）
├── _screenshots/                           Step 3 验证截图（22 段 + 全页）
├── assets/
│   ├── vendor/echarts.min.js              离线 ECharts (~1 MB)
│   ├── css/{tokens,base,layout,print}.css
│   └── js/{charts,nav}.js
├── scripts/
│   └── export-pdf.mjs                      Playwright 导出脚本（A3 横版）
└── dist/
    ├── nlp-voc-report.pdf                  主版 · 22 页 · 不含 speaker notes
    └── nlp-voc-report-with-notes.pdf       附录版 · 44 页 · 每段后附 speaker notes
```

## 本地预览

```bash
cd paper2skills-vault/07-NLP-VOC/presentation
python3 -m http.server 8765
open http://localhost:8765/index.html
```

## 重新导出 PDF

```bash
cd paper2skills-vault/07-NLP-VOC/presentation
npm i --no-audit --no-fund playwright
node scripts/export-pdf.mjs
```

脚本会临时起一个 `127.0.0.1:8766` server，分别用 `data-pdf="main"` 与 `data-pdf="notes"` 模式渲染 PDF，输出到 `dist/`。

## PDF 设计契约

- A3 横版（420 × 297 mm），12mm 页边距
- 每个 `.section` 强制独立分页
- 主版隐藏所有 `aside.notes`
- 附录版每段后插一页 speaker notes（左侧 4px 边线 + 浅灰底）
- 全程颜色保真（`print-color-adjust: exact`）

## 设计锁定（贯穿 Step 1-4）

- 风格：白底咨询公司风
- 主色：`#1E40AF`（数据 / AI）/ `#0F766E`（产品 / 平台）/ `#B45309`（成果 / 数字）
- 字体：PingFang SC + Inter
- 数据口径：364,569
- 看板素材：示意 mockup，明确「示意 · MOCKUP」标识
- 风险披露：主线不展开，集中放附录 A2

## 后续可选 Step 5（待你确认）

- 接入真实 Superset dashboard 截图替换 mockup
- 增加封面 Logo / 公司品牌资产
- 钉钉 / 飞书可分享 PDF 链接配置
- 演练版（含计时与 cue point）
