/* NLP-VOC Presentation · Charts (ECharts)
 * 5 instances driven by data attributes / ids on the page.
 *   chart-coverage         · §02 / §13
 *   chart-gate             · §02 / §13
 *   chart-data-source      · §04
 *   chart-precision-fix    · §07
 *   chart-cost-quality     · §15
 */

(function () {
  if (!window.echarts) {
    console.warn("[charts] ECharts not loaded; skipping render.");
    return;
  }

  const palette = {
    ink: "#111827",
    ink2: "#374151",
    ink3: "#6b7280",
    divider: "#e5e7eb",
    data: "#1e40af",
    dataSoft: "#dbeafe",
    platform: "#0f766e",
    platformSoft: "#ccfbf1",
    accent: "#b45309",
    accentSoft: "#fef3c7",
    risk: "#b91c1c",
    riskSoft: "#fee2e2",
  };

  const ff = "Inter, IBM Plex Sans, -apple-system, BlinkMacSystemFont, sans-serif";

  const baseGrid = { left: 56, right: 24, top: 32, bottom: 32, containLabel: true };
  const baseAxis = {
    axisLine: { lineStyle: { color: palette.divider } },
    axisTick: { show: false },
    axisLabel: { color: palette.ink3, fontSize: 12, fontFamily: ff },
    splitLine: { lineStyle: { color: palette.divider, type: "dashed" } },
  };
  const baseTooltip = {
    trigger: "axis",
    confine: true,
    backgroundColor: "#fff",
    borderColor: palette.divider,
    textStyle: { color: palette.ink, fontFamily: ff, fontSize: 12 },
    extraCssText: "box-shadow: 0 8px 24px rgba(17,24,39,0.06)",
  };

  const charts = [];
  const init = (id) => {
    const el = document.getElementById(id);
    if (!el) return null;
    const c = echarts.init(el, null, { renderer: "svg" });
    charts.push(c);
    return c;
  };
  const initAll = (selector, builder) => {
    document.querySelectorAll(selector).forEach((el) => {
      const c = echarts.init(el, null, { renderer: "svg" });
      c.setOption(builder(el));
      charts.push(c);
    });
  };

  /* ---------- §02 / §13 · 覆盖率提升 ---------- */
  initAll("[data-chart='coverage']", () => ({
    grid: baseGrid,
    tooltip: baseTooltip,
    xAxis: { type: "category", data: ["Phase 4 基线", "Phase 5 D2", "Phase 5 D7"], ...baseAxis },
    yAxis: {
      type: "value", min: 80, max: 100,
      axisLabel: { ...baseAxis.axisLabel, formatter: "{value}%" },
      splitLine: baseAxis.splitLine,
      axisLine: baseAxis.axisLine,
      axisTick: baseAxis.axisTick,
    },
    series: [
      {
        name: "5K 子集覆盖率",
        type: "bar", barWidth: 36,
        data: [82.58, 95.4, 97.22],
        itemStyle: { color: palette.data, borderRadius: [2, 2, 0, 0] },
        label: { show: true, position: "top", color: palette.ink, fontFamily: ff, fontSize: 13, formatter: "{c}%" },
      },
      {
        type: "line", showSymbol: false,
        markLine: {
          silent: true, symbol: "none",
          label: { color: palette.ink3, fontSize: 11, formatter: "目标 95%" },
          lineStyle: { color: palette.accent, type: "dashed" },
          data: [{ yAxis: 95 }],
        },
      },
    ],
  }));

  /* ---------- §02 / §13 · Gate × Precision 演进 ---------- */
  initAll("[data-chart='gate']", () => ({
    grid: { ...baseGrid, top: 48 },
    tooltip: baseTooltip,
    legend: { right: 0, top: 0, textStyle: { color: palette.ink3, fontSize: 12, fontFamily: ff }, itemWidth: 12, itemHeight: 8 },
    xAxis: { type: "category", data: ["P5·D13", "P5·D14", "P6·D3", "P6·D5", "P6·D8", "P6·D9"], ...baseAxis },
    yAxis: [
      { type: "value", name: "Gate 通过项", nameTextStyle: { color: palette.ink3, fontSize: 12, fontFamily: ff }, min: 0, max: 7, interval: 1, ...baseAxis },
      { type: "value", name: "Precision", nameTextStyle: { color: palette.ink3, fontSize: 12, fontFamily: ff }, min: 0.6, max: 1, ...baseAxis, axisLabel: { ...baseAxis.axisLabel, formatter: "{value}" } },
    ],
    series: [
      {
        name: "Week 2 Gate (n/7)", type: "bar", barWidth: 28,
        data: [4, 4, 5, 7, 5, 7],
        itemStyle: { color: palette.platform, borderRadius: [2, 2, 0, 0] },
        label: { show: true, position: "top", color: palette.ink, fontFamily: ff, fontSize: 12 },
      },
      {
        name: "Precision (口径 B)", type: "line", yAxisIndex: 1,
        symbol: "circle", symbolSize: 8,
        data: [null, null, null, null, 0.885, 0.896],
        lineStyle: { color: palette.accent, width: 2 },
        itemStyle: { color: palette.accent },
        label: { show: true, position: "top", color: palette.accent, fontFamily: ff, fontSize: 12, formatter: "{c}" },
      },
    ],
  }));

  /* ---------- §04 · 5 数据源占比 (横向堆叠) ---------- */
  initAll("[data-chart='data-source']", () => {
    const sources = [
      { name: "Amazon competitor", value: 194734, color: palette.data },
      { name: "Trustpilot",        value: 99853,  color: "#3b82f6" },
      { name: "Zendesk",           value: 47204,  color: palette.platform },
      { name: "Momcozy",           value: 19808,  color: "#14b8a6" },
      { name: "Reddit",            value: 2970,   color: palette.accent },
    ];
    const total = sources.reduce((s, x) => s + x.value, 0);
    return {
      grid: { left: 8, right: 8, top: 16, bottom: 8, containLabel: false },
      tooltip: {
        trigger: "item",
        backgroundColor: "#fff", borderColor: palette.divider,
        textStyle: { color: palette.ink, fontFamily: ff, fontSize: 12 },
        formatter: (p) => `${p.seriesName}<br/>${p.value.toLocaleString()} 条 · ${(p.value / total * 100).toFixed(1)}%`,
      },
      xAxis: { type: "value", show: false, max: total },
      yAxis: { type: "category", show: false, data: ["VOC"] },
      series: sources.map((s) => ({
        name: s.name, type: "bar", stack: "total", barWidth: 56,
        data: [s.value], itemStyle: { color: s.color },
        label: {
          show: true, position: "inside",
          color: "#fff", fontFamily: ff, fontSize: 12,
          lineHeight: 16,
          formatter: () => `${s.name}\n${(s.value / total * 100).toFixed(1)}%`,
        },
      })),
    };
  });

  /* ---------- §07 · Precision 修复 Timeline ---------- */
  initAll("[data-chart='precision-fix']", () => ({
    grid: { ...baseGrid, top: 48 },
    tooltip: baseTooltip,
    legend: { right: 0, top: 0, textStyle: { color: palette.ink3, fontSize: 12, fontFamily: ff }, itemWidth: 12, itemHeight: 8 },
    xAxis: { type: "category", data: ["D7 Spot Check", "D8 Strict Prompt", "D9 Method C"], ...baseAxis },
    yAxis: [
      { type: "value", name: "Precision (口径 B)", nameTextStyle: { color: palette.ink3, fontSize: 12, fontFamily: ff }, min: 0.5, max: 1, ...baseAxis },
      { type: "value", name: "Gate (n/7)", nameTextStyle: { color: palette.ink3, fontSize: 12, fontFamily: ff }, min: 0, max: 7, interval: 1, ...baseAxis },
    ],
    series: [
      {
        name: "Precision", type: "line", smooth: false,
        symbol: "circle", symbolSize: 10,
        data: [0.639, 0.885, 0.896],
        lineStyle: { color: palette.accent, width: 3 },
        itemStyle: { color: palette.accent },
        label: { show: true, position: "top", color: palette.accent, fontFamily: ff, fontSize: 13, formatter: "{c}" },
      },
      {
        name: "Week 2 Gate", type: "bar", yAxisIndex: 1, barWidth: 24,
        data: [7, 5, 7],
        itemStyle: { color: palette.platform, borderRadius: [2, 2, 0, 0], opacity: 0.6 },
        label: { show: true, position: "top", color: palette.platform, fontFamily: ff, fontSize: 12 },
      },
    ],
  }));

  /* ---------- §15 · 成本 × 质量象限 ---------- */
  initAll("[data-chart='cost-quality']", () => ({
    grid: { left: 64, right: 56, top: 32, bottom: 56, containLabel: true },
    tooltip: {
      trigger: "item",
      backgroundColor: "#fff", borderColor: palette.divider,
      textStyle: { color: palette.ink, fontFamily: ff, fontSize: 12 },
      formatter: (p) => `${p.data.name}<br/>成本：${p.data.costLabel}<br/>质量：${p.data.qualityLabel}`,
    },
    xAxis: {
      type: "value", name: "成本 (相对)", nameLocation: "middle", nameGap: 28,
      nameTextStyle: { color: palette.ink3, fontSize: 12, fontFamily: ff },
      min: 0, max: 100,
      axisLabel: { ...baseAxis.axisLabel, formatter: "{value}" },
      splitLine: baseAxis.splitLine, axisLine: baseAxis.axisLine,
    },
    yAxis: {
      type: "value", name: "质量 (覆盖 × 精度)", nameLocation: "middle", nameGap: 48,
      nameTextStyle: { color: palette.ink3, fontSize: 12, fontFamily: ff },
      min: 0, max: 100,
      axisLabel: { ...baseAxis.axisLabel, formatter: "{value}" },
      splitLine: baseAxis.splitLine, axisLine: baseAxis.axisLine,
    },
    series: [
      {
        type: "scatter", symbolSize: 24,
        data: [
          { name: "Phase 5/6/7", value: [6, 92], costLabel: "≈¥150 + 7h 工程", qualityLabel: "覆盖 97.22% · precision 0.896",
            itemStyle: { color: palette.data }, label: { show: true, formatter: "● Phase 5/6/7", position: "right", color: palette.data, fontFamily: ff, fontSize: 13, fontWeight: 600 } },
          { name: "Phase 4 规则", value: [3, 55], costLabel: "≈¥0 + 历史人力", qualityLabel: "覆盖 82.58% · 无精度门禁",
            itemStyle: { color: palette.ink3 }, label: { show: true, formatter: "Phase 4 规则", position: "right", color: palette.ink3, fontFamily: ff, fontSize: 12 } },
          { name: "GPT-4 直调", value: [82, 88], costLabel: "≈¥54,000", qualityLabel: "覆盖与精度高，但成本不可控",
            itemStyle: { color: palette.accent }, label: { show: true, formatter: "GPT-4 直调", position: "right", color: palette.accent, fontFamily: ff, fontSize: 12 } },
          { name: "外包人工", value: [92, 70], costLabel: "≈¥360,000 + 3-6 月", qualityLabel: "受人影响，不稳定",
            itemStyle: { color: palette.risk }, label: { show: true, formatter: "外包人工", position: "right", color: palette.risk, fontFamily: ff, fontSize: 12 } },
        ],
        markLine: {
          silent: true, symbol: "none",
          lineStyle: { color: palette.divider, type: "dashed" },
          label: { show: false },
          data: [{ xAxis: 50 }, { yAxis: 50 }],
        },
      },
    ],
  }));

  window.addEventListener("resize", () => charts.forEach((c) => c.resize()));
})();
