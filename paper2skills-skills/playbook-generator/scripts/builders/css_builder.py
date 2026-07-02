"""CSS asset builder for paper2skills Playbook."""
from __future__ import annotations


def build_css() -> str:
    return """
/* ═══════════════════════════════════════════════════════
   paper2skills Playbook — Design System v5
   Smartisan Spirit · Restrained Tech · Linear Precision
   ═══════════════════════════════════════════════════════ */

:root {
  --bg:          #F6F6F6;
  --bg-warm:     #F0F0F0;
  --panel:       #FFFFFF;
  --panel-2:     #F3F3F3;
  --panel-3:     #ECECEC;

  --ink:         #1A1A2E;
  --ink-2:       #3D3D52;
  --muted:       #888888;

  --line:        #E4E4E4;
  --line-strong: #CCCCCC;

  --accent:      #B5323E;
  --accent-dark: #8C2530;
  --accent-light:#FDF0F1;
  --accent-bg:   #FDF0F1;
  --accent2:     #555555;
  --accent2-bg:  #F0F0F0;

  --green:       #16A34A;
  --green-bg:    #F0FDF4;
  --green-dark:  #14532D;
  --amber:       #D97706;
  --amber-bg:    #FFFBEB;
  --amber-dark:  #92400E;
  --red:         #DC2626;
  --red-bg:      #FEF2F2;

  --phase-1:     #B5323E; --phase-1-bg: #FDF0F1; --phase-1-muted: #c86870;
  --phase-2:     #555555; --phase-2-bg: #F0F0F0; --phase-2-muted: #999999;
  --phase-3:     #14532D; --phase-3-bg: #F0FDF4; --phase-3-muted: #4a9466;

  --tag-bg:      #EFEFEF;
  --tag-ink:     #444444;
  --tag-topic-bg:#F0FDF4;
  --tag-topic-ink:#14532D;

  --nav-bg:         #1C2B3A;
  --nav-border:     #2A3D52;
  --nav-text:       #8FA8BE;
  --nav-text-hover: #D8E6F0;
  --nav-active-bg:  transparent;
  --nav-active-text:#FFFFFF;
  --nav-highlight:  #B5323E;
  --topbar-height:  52px;

  --sidebar-bg:  #FAFAFA;
  --sidebar-w:   220px;

  /* 间距系统 - 8px网格 */
  --sp-1: 4px; --sp-2: 8px; --sp-3: 12px; --sp-4: 16px;
  --sp-5: 20px; --sp-6: 24px; --sp-8: 32px; --sp-10: 40px;
  --sp-12: 48px; --sp-16: 64px;

  /* 字体规模 */
  --text-xs: 11px; --text-sm: 12.5px; --text-base: 14px;
  --text-md: 15px; --text-lg: 17px; --text-xl: 20px;
  --text-2xl: 24px; --text-3xl: 32px; --text-4xl: 40px;
  --text-5xl: 52px;
  --lh-tight: 1.2; --lh-snug: 1.4; --lh-base: 1.65; --lh-relaxed: 1.8;

  /* 数字字体 */
  --font-num: "SF Mono", "JetBrains Mono", ui-monospace, monospace;

  --r-xs:  2px;
  --r-sm:  4px;
  --r-md:  6px;
  --r-lg:  8px;
  --r-xl:  10px;
  --r-2xl: 12px;
  --r-full:999px;

  /* 渐变系统 */
  --grad-accent: linear-gradient(135deg, #B5323E 0%, #8C2530 100%);
  --grad-dark: linear-gradient(135deg, #1C2B3A 0%, #111827 100%);
  --grad-surface: linear-gradient(180deg, #ffffff 0%, #f8f8f8 100%);
  --grad-glow: radial-gradient(ellipse at 50% 0%, rgba(181,50,62,0.12) 0%, transparent 70%);

  /* 阴影升级 */
  --shadow-xs:    0 1px 2px rgba(0,0,0,0.04);
  --shadow-sm:    0 2px 4px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
  --shadow-md:    0 4px 12px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04);
  --shadow-lg:    0 12px 28px rgba(0,0,0,0.10), 0 4px 8px rgba(0,0,0,0.06);
  --shadow-xl:    0 24px 48px rgba(0,0,0,0.14), 0 8px 16px rgba(0,0,0,0.08);
  --shadow-accent: 0 6px 20px rgba(181,50,62,0.25), 0 2px 6px rgba(181,50,62,0.15);
  --shadow-inner:  inset 0 1px 3px rgba(0,0,0,0.06);

  /* 过渡升级 */
  --t: .15s cubic-bezier(0.4, 0, 0.2, 1);
  --t-card: .2s cubic-bezier(0.34, 1.56, 0.64, 1);
  --t-slow: .3s cubic-bezier(0.4, 0, 0.2, 1);
  --t-spring: .35s cubic-bezier(0.34, 1.56, 0.64, 1);

  --font: "Helvetica Neue", -apple-system, "SF Pro Display", BlinkMacSystemFont,
          "PingFang SC", "Hiragino Sans GB", "Noto Sans SC", "Microsoft YaHei",
          sans-serif;
}

@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
  from { opacity: 0; }
  to   { opacity: 1; }
}
@keyframes slideInRight {
  from { opacity: 0; transform: translateX(24px); }
  to   { opacity: 1; transform: translateX(0); }
}
@keyframes scaleIn {
  from { opacity: 0; transform: scale(0.95); }
  to   { opacity: 1; transform: scale(1); }
}
@keyframes shimmer {
  0%   { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
@keyframes countUp {
  from { opacity: 0; transform: translateY(8px) scale(0.9); }
  to   { opacity: 1; transform: translateY(0) scale(1); }
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50%       { opacity: .5; }
}
@keyframes blink {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0; }
}
@keyframes spin {
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
}
.anim-fade-in-up { animation: fadeInUp .4s ease both; }
.anim-fade-in    { animation: fadeIn .3s ease both; }
.anim-scale-in   { animation: scaleIn .25s ease both; }
.anim-delay-1    { animation-delay: .05s; }
.anim-delay-2    { animation-delay: .10s; }
.anim-delay-3    { animation-delay: .15s; }
.anim-delay-4    { animation-delay: .20s; }
.anim-delay-5    { animation-delay: .25s; }

*, *::before, *::after { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: var(--font);
  font-size: 14px;
  line-height: 1.65;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  letter-spacing: -.01em;
}
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.card:hover, .skill-card:hover, .biz-card:hover, .domain-card:hover,
.metric-card:hover, .wf-card:hover, .agent-card:hover, .ds-card:hover,
a.card:hover, a.skill-card:hover, a.biz-card:hover, a.domain-card:hover,
a.metric-card:hover, a.wf-card:hover { text-decoration: none; }
img { max-width: 100%; }
strong { font-weight: 600; }
p { margin: 0 0 12px; }
p:last-child { margin-bottom: 0; }

.topbar {
  position: sticky; top: 0; z-index: 200;
  display: flex; align-items: center;
  height: var(--topbar-height);
  padding: 0 20px 0 0;
  background: var(--nav-bg);
  border-bottom: 1px solid rgba(255,255,255,0.08);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
}
.brand {
  display: flex; align-items: center; gap: 10px;
  text-decoration: none; color: #EEEEEE;
  flex-shrink: 0; padding: 0 18px;
  height: 100%; border-right: 1px solid var(--nav-border);
}
.brand:hover { text-decoration: none; }
.brand-icon {
  width: 24px; height: 24px; border-radius: var(--r-sm);
  background: var(--accent); color: #fff;
  font-weight: 800; font-size: 11px;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0; letter-spacing: -.02em;
}
.brand-name { font-weight: 700; font-size: 13.5px; letter-spacing: -.02em; color: #EEEEEE; line-height: 1; }
.brand-tag { display: block; font-size: 10px; font-weight: 400; color: #555555; letter-spacing: .05em; text-transform: uppercase; margin-top: 2px; }
.topbar-right { margin-left: auto; display: flex; align-items: center; gap: 10px; }
#global-search {
  width: min(260px, 22vw); padding: 6px 12px 6px 30px;
  border-radius: var(--r-sm); border: 1px solid #2E2E2E;
  background: #1A1A1A; color: #CCCCCC; font-size: 12.5px;
  font-family: var(--font);
  transition: border-color var(--t), background var(--t);
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%23555' stroke-width='2'%3E%3Ccircle cx='11' cy='11' r='8'/%3E%3Cpath d='m21 21-4.35-4.35'/%3E%3C/svg%3E");
  background-repeat: no-repeat; background-position: 10px center;
}
#global-search::placeholder { color: #555555; opacity: 1; }
#global-search:hover { border-color: #3E3E3E; background: #202020; }
#global-search:focus { outline: none; border-color: #555555; background: #1E1E1E; color: #EEEEEE; }
.topbar-cta {
  display: inline-flex; align-items: center;
  padding: 7px 16px; border-radius: var(--r-sm);
  background: var(--accent); color: #fff !important;
  font-size: 13px; font-weight: 600; letter-spacing: -.01em;
  text-decoration: none !important;
  transition: background var(--t); white-space: nowrap; flex-shrink: 0;
}
.topbar-cta:hover { background: var(--accent-dark); }
.topbar-cta.active { background: var(--accent-dark); }

.topbar-nav {
  display: flex; align-items: center; height: 100%;
  gap: 0; margin-left: 4px; flex-shrink: 0;
}
.topbar-nav a {
  display: inline-flex; align-items: center;
  height: 100%; padding: 0 14px;
  font-size: 13px; font-weight: 500;
  color: var(--nav-text);
  text-decoration: none;
  letter-spacing: -.01em;
  border-bottom: 2px solid transparent;
  transition: color var(--t), border-color var(--t), background var(--t);
  white-space: nowrap;
}
.topbar-nav a:hover {
  color: var(--nav-text-hover);
  background: rgba(255,255,255,.05);
  text-decoration: none;
  border-bottom-color: rgba(255,255,255,.15);
}
.topbar-nav a.active {
  color: #fff;
  font-weight: 600;
  border-bottom-color: var(--accent);
}
.topbar-brand {
  display: inline-flex; align-items: center;
  height: 100%; padding: 0 18px;
  font-size: 14px; font-weight: 700;
  color: #EEEEEE; text-decoration: none;
  letter-spacing: -.03em;
  border-right: 1px solid var(--nav-border);
  flex-shrink: 0; white-space: nowrap;
  transition: color var(--t);
}
.topbar-brand:hover { color: #fff; text-decoration: none; }
.topbar-stat {
  font-size: 11.5px; color: var(--nav-text);
  font-weight: 500; letter-spacing: .01em;
  white-space: nowrap;
  padding: 4px 10px; border-radius: var(--r-sm);
  background: rgba(255,255,255,.06);
  border: 1px solid rgba(255,255,255,.08);
}

.hamburger {
  display: none; flex-direction: column; justify-content: center;
  gap: 5px; width: 40px; height: var(--topbar-height); padding: 0 10px;
  background: none; border: none; border-right: 1px solid var(--nav-border);
  cursor: pointer; flex-shrink: 0;
}
.hamburger span { display: block; height: 1.5px; background: #666666; border-radius: 2px; transition: transform var(--t), opacity var(--t); }
.hamburger.open span:nth-child(1) { transform: translateY(6.5px) rotate(45deg); }
.hamburger.open span:nth-child(2) { opacity: 0; }
.hamburger.open span:nth-child(3) { transform: translateY(-6.5px) rotate(-45deg); }
.hamburger:hover span { background: #AAAAAA; }

.layout { display: grid; grid-template-columns: 220px 1fr; min-height: calc(100vh - var(--topbar-height)); }

.sidebar {
  display: flex; flex-direction: column;
  background: var(--panel);
  border-right: 1px solid var(--line);
  position: sticky; top: var(--topbar-height);
  height: calc(100vh - var(--topbar-height));
  overflow-y: auto; overflow-x: hidden;
}
.sidebar::-webkit-scrollbar { width: 3px; }
.sidebar::-webkit-scrollbar-track { background: transparent; }
.sidebar::-webkit-scrollbar-thumb { background: var(--line-strong); border-radius: 2px; }
.sb-top { flex: 1; padding: 8px 6px 6px; display: flex; flex-direction: column; gap: 0; overflow-y: auto; overflow-x: hidden; }
.sb-section {
  margin-bottom: 0;
  padding-bottom: 4px;
}
.sb-section + .sb-section {
  border-top: 1px solid var(--line);
  padding-top: 4px;
  margin-top: 4px;
}
.sb-label {
  font-size: 9.5px; font-weight: 700; letter-spacing: .10em;
  text-transform: uppercase; color: var(--muted);
  padding: 8px 8px 3px; margin: 0; user-select: none;
  opacity: .7;
}
.sb-links { display: flex; flex-direction: column; gap: 1px; }
.sidebar a {
  display: flex; align-items: center; gap: 8px;
  color: var(--ink-2); text-decoration: none;
  padding: 6px 8px; border-radius: var(--r-sm);
  font-size: 12.5px; font-weight: 400; line-height: 1.3;
  transition: background var(--t), color var(--t);
  position: relative;
}
.sidebar a:hover { background: var(--panel-2); color: var(--ink); text-decoration: none; }
.sidebar a.active, .sidebar a[aria-current="page"] { background: rgba(181,50,62,0.06); color: var(--accent); font-weight: 600; }
.sidebar a.active::before, .sidebar a[aria-current="page"]::before {
  content: ''; position: absolute; left: 0; top: 5px; bottom: 5px;
  width: 2px; border-radius: 0 2px 2px 0; background: var(--accent);
}
.sbl-icon { width: 16px; height: 16px; display: flex; align-items: center; justify-content: center; font-size: 10px; flex-shrink: 0; opacity: 0.4; transition: opacity var(--t); }
.sidebar a:hover .sbl-icon, .sidebar a.active .sbl-icon, .sidebar a[aria-current="page"] .sbl-icon { opacity: 1; }
.sbl-text { flex: 1; }
.sb-bottom {
  flex-shrink: 0;
  padding: 10px 8px 12px;
  border-top: 1px solid var(--line);
  background: var(--panel);
}
.sb-upgrade-card {
  display: flex; flex-direction: column; gap: 3px;
  padding: 9px 11px;
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%);
  border-radius: var(--r-lg);
  margin-bottom: 5px;
  text-decoration: none;
  transition: opacity var(--t), transform var(--t);
}
.sb-upgrade-card:hover { opacity: .92; transform: translateY(-1px); text-decoration: none; }
.sb-upgrade-label {
  font-size: 9px; font-weight: 700; letter-spacing: .1em;
  text-transform: uppercase; color: rgba(255,255,255,.65);
}
.sb-upgrade-title {
  font-size: 11.5px; font-weight: 700; color: #fff; line-height: 1.25;
}
.sb-upgrade-sub {
  font-size: 10px; color: rgba(255,255,255,.7); line-height: 1.4;
}
.sb-settings-link {
  display: flex; align-items: center; gap: 7px;
  padding: 6px 8px; border-radius: var(--r-sm);
  font-size: 12px; color: var(--muted); text-decoration: none;
  transition: background var(--t), color var(--t);
}
.sb-settings-link:hover { background: var(--panel-2); color: var(--ink); text-decoration: none; }
.sb-settings-icon { font-size: 13px; opacity: .6; flex-shrink: 0; }

.content { padding: 30px 40px; max-width: 1400px; overflow-x: hidden; }
.mobile-nav-overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.6); backdrop-filter: blur(4px); z-index: 190; }
.mobile-nav-overlay.show { display: block; }

.content h1 { font-size: 34px; font-weight: 800; letter-spacing: -.05em; line-height: 1.05; margin: 0 0 10px; color: var(--ink); }
.content h2 { font-size: 17px; font-weight: 700; letter-spacing: -.03em; margin: 24px 0 10px; color: var(--ink); line-height: 1.3; }
.content h2:first-child { margin-top: 0; }
.content h3 { font-size: 15px; font-weight: 600; letter-spacing: -.02em; margin: 20px 0 8px; color: var(--ink); line-height: 1.3; }
.content h4 { font-size: 13px; font-weight: 600; letter-spacing: -.01em; margin: 16px 0 6px; color: var(--ink-2); }
.lead { font-size: 15px; color: var(--muted); margin: 0 0 20px; line-height: 1.7; letter-spacing: -.01em; }
.muted { color: var(--muted); }
.section-eyebrow { font-size: 11px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin: 0 0 6px; display: block; }
.section-head { margin-bottom: 20px; }
.section-head h2 { margin: 0 0 4px; border: none; padding: 0; }
.section-head p { margin: 0; font-size: 13.5px; color: var(--muted); line-height: 1.6; }

.hero {
  padding: 56px 0 48px;
  text-align: center;
  position: relative;
  overflow: hidden;
}
.hero::before {
  content: '';
  position: absolute;
  top: -120px; left: 50%;
  transform: translateX(-50%);
  width: 800px; height: 400px;
  background: radial-gradient(ellipse, rgba(181,50,62,0.08) 0%, transparent 70%);
  pointer-events: none;
}
.hero-badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 14px;
  background: rgba(181,50,62,0.08);
  border: 1px solid rgba(181,50,62,0.2);
  border-radius: var(--r-full);
  font-size: 11.5px; font-weight: 600;
  color: var(--accent); letter-spacing: .01em;
  margin-bottom: 24px;
  animation: fadeInUp .4s ease both;
}
.hero h1 {
  font-size: clamp(24px, 4vw, 42px);
  font-weight: 800;
  line-height: 1.15;
  letter-spacing: -.04em;
  color: var(--ink);
  margin: 0 auto 20px;
  max-width: 800px;
  animation: fadeInUp .4s .05s ease both;
}
.hero .lead {
  font-size: 15px; line-height: 1.7;
  color: var(--ink-2); max-width: 600px;
  margin: 0 auto 32px;
  animation: fadeInUp .4s .1s ease both;
}
.hero-primary-cta {
  display: flex; flex-wrap: wrap; gap: 10px;
  justify-content: center; align-items: center;
  animation: fadeInUp .4s .15s ease both;
}
.hero-tabs { display: flex; gap: 5px; flex-wrap: wrap; margin-bottom: 20px; }
.tab-btn {
  padding: 7px 18px; border: 1px solid var(--line-strong);
  background: var(--panel); border-radius: var(--r-sm);
  font-size: 12.5px; font-weight: 500; font-family: var(--font);
  cursor: pointer; color: var(--muted);
  transition: background var(--t), color var(--t), border-color var(--t);
}
.tab-btn:hover:not(.active) { background: var(--panel-2); color: var(--ink); border-color: var(--line-strong); }
.tab-btn.active { background: var(--ink); border-color: var(--ink); color: #fff; font-weight: 600; }
.tab-btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.tab-panel { display: none; }
.tab-panel.active { display: block; }

.btn-primary, .topbar-cta {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 10px 22px;
  border-radius: var(--r-md);
  font-size: 13.5px; font-weight: 600;
  letter-spacing: -.01em;
  text-decoration: none !important;
  white-space: nowrap; flex-shrink: 0;
  transition: all var(--t);
  position: relative; overflow: hidden;
}
.btn-primary, a.btn-primary, .topbar-cta {
  background: var(--grad-accent);
  color: #fff !important;
  box-shadow: var(--shadow-accent);
  border: none;
}
.btn-primary:hover, .topbar-cta:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 24px rgba(181,50,62,0.32), 0 3px 8px rgba(181,50,62,0.20);
  background: linear-gradient(135deg, #c23845 0%, #9a2b38 100%);
}
.btn-primary:active { transform: translateY(0); }
.btn-primary:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }

.btn-secondary {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 10px 22px;
  border-radius: var(--r-md);
  font-size: 13.5px; font-weight: 600;
  background: var(--panel);
  color: var(--ink) !important;
  border: 1.5px solid var(--line-strong);
  box-shadow: var(--shadow-sm);
  text-decoration: none !important;
  transition: all var(--t);
}
.btn-secondary:hover {
  border-color: var(--line-strong);
  box-shadow: var(--shadow-md);
  transform: translateY(-1px);
  text-decoration: none;
}
.btn-secondary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

.btn-sm { padding: 6px 14px; font-size: 12.5px; }
.btn-lg { padding: 13px 28px; font-size: 15px; border-radius: var(--r-lg); }

.icon-badge {
  display: inline-flex; align-items: center; justify-content: center;
  width: 38px; height: 38px; background: var(--panel-3); color: var(--ink-2);
  border-radius: var(--r-md); font-size: 10px; font-weight: 700;
  letter-spacing: .04em; flex-shrink: 0; font-family: var(--font);
  text-transform: uppercase; border: 1px solid var(--line);
}
.icon-badge.warm { background: var(--amber-bg); color: var(--amber-dark); }
.icon-badge.green { background: var(--green-bg); color: var(--green-dark); }
.icon-badge.red { background: var(--red-bg); color: var(--red); }
.icon-badge.dark { background: var(--panel-3); color: var(--ink); }
.icon-badge.lg { width: 42px; height: 42px; font-size: 11px; border-radius: var(--r-lg); }

.biz-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; margin: 14px 0; align-items: stretch; }
.biz-card {
  display: flex; flex-direction: column; background: var(--panel);
  border: 1px solid var(--line); border-radius: var(--r-lg);
  padding: 18px; text-decoration: none; color: var(--ink); box-shadow: none;
  transition: border-color var(--t-card), box-shadow var(--t-card), transform var(--t-card);
}
.biz-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); border-color: var(--line-strong); text-decoration: none; }
.biz-card-header { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.biz-icon {
  width: 40px; height: 40px; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  background: var(--panel-3); color: var(--ink-2);
  border-radius: var(--r-lg); overflow: hidden;
  border: 1px solid var(--line);
}
.biz-icon svg { width: 20px; height: 20px; }
.biz-card-meta { margin-bottom: 5px; }
.biz-body { flex: 1; min-width: 0; display: flex; flex-direction: column; }
.biz-body strong { display: block; font-size: 13.5px; font-weight: 600; letter-spacing: -.02em; line-height: 1.35; color: var(--ink); word-break: keep-all; overflow-wrap: break-word; }
.biz-body p { margin: 6px 0 0; font-size: 12.5px; color: var(--muted); line-height: 1.6; flex: 1; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.biz-card-footer { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 10px; padding-top: 9px; border-top: 1px solid var(--line); }
.biz-tag { display: inline-flex; align-items: center; flex-shrink: 0; font-size: 10px; font-weight: 500; background: var(--panel-2); color: var(--ink-2); padding: 2px 7px; border-radius: var(--r-xs); white-space: nowrap; border: 1px solid var(--line); }

.ds-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 12px; margin: 12px 0; align-items: stretch; }
.ds-card { background: var(--panel); border: 1px solid var(--line); border-radius: var(--r-lg); padding: 16px; box-shadow: none; }
.ds-card h3 { margin: 0 0 8px; font-size: 13px; font-weight: 700; }
.hot-list { padding: 0; margin: 5px 0 0; list-style: none; display: flex; flex-direction: column; gap: 4px; }
.hot-list li { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
.hot-list a { color: var(--ink-2); text-decoration: none; flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 12px; }
.hot-list a:hover { color: var(--accent); }
.hot-list .roi-badge { flex-shrink: 0; }
.algo-tags { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 8px; }
.algo-tags .tag { text-decoration: none; }
.ceo-entry { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin: 16px 0; align-items: start; }
.ceo-entry-body h3 { margin: 0 0 8px; font-size: 16px; font-weight: 700; }
.ceo-entry-body p { color: var(--muted); margin: 0 0 14px; font-size: 13px; line-height: 1.6; }
.ceo-phases { display: flex; flex-direction: column; gap: 8px; }
.ceo-phase { background: var(--panel); border-left: 2px solid; border-radius: 0 var(--r-md) var(--r-md) 0; padding: 10px 14px; font-size: 13px; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); border-right: 1px solid var(--line); }
.ceo-phase p { margin: 3px 0; color: var(--muted); }

.metrics {
  display: flex; flex-wrap: wrap; gap: 24px;
  margin: 24px 0;
}
.metrics div {
  display: flex; flex-direction: column; gap: 2px;
  min-width: 100px;
  background: var(--panel); border: 1px solid var(--line); border-radius: var(--r-lg); padding: 16px;
}
.metrics div strong, .metrics strong {
  font-size: 32px; font-weight: 800;
  font-family: var(--font-num, var(--font));
  letter-spacing: -.05em; line-height: 1;
  color: var(--ink);
  animation: countUp .5s ease both;
  margin-bottom: 3px; display: block;
}
.metrics div strong.accent-num { color: var(--accent); }
.metrics div span, .metrics span {
  font-size: 12px; color: var(--muted); font-weight: 500;
  letter-spacing: .02em; text-transform: uppercase;
}

.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; margin: 14px 0; align-items: stretch; }
.metric-card, .domain-card {
  display: flex; flex-direction: column; background: var(--panel);
  border: 1px solid var(--line); border-radius: var(--r-lg);
  padding: 14px 16px; text-decoration: none; color: var(--ink); min-height: 60px;
  transition: border-color var(--t-card), box-shadow var(--t-card), transform var(--t-card);
}
.metric-card:hover, .domain-card:hover { transform: translateY(-1px); box-shadow: var(--shadow-sm); border-color: var(--line-strong); text-decoration: none; }
.metric-card strong { display: block; font-weight: 600; font-size: 13px; letter-spacing: -.02em; }
.metric-card span { color: var(--muted); font-size: 11.5px; }
.domain-card strong { font-size: 13px; font-weight: 600; letter-spacing: -.02em; color: var(--ink); display: block; margin-bottom: 2px; }
.domain-card span { font-size: 11.5px; color: var(--muted); }

.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; margin: 12px 0; align-items: stretch; }
.skill-card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--r-lg);
  padding: 18px;
  display: flex; flex-direction: column;
  gap: 8px;
  transition: all var(--t-card);
  position: relative;
  overflow: hidden;
  color: var(--ink); text-decoration: none;
}
.skill-card::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0;
  height: 2px;
  background: var(--grad-accent);
  opacity: 0;
  transition: opacity var(--t);
}
.skill-card:hover {
  transform: translateY(-3px);
  box-shadow: var(--shadow-lg);
  border-color: rgba(181,50,62,0.2);
  text-decoration: none;
}
.skill-card:hover::before { opacity: 1; }
.skill-card h3 { margin: 0; font-size: 13.5px; font-weight: 600; letter-spacing: -.02em; line-height: 1.35; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.skill-card h3 a { color: var(--ink); text-decoration: none; }
.skill-card h3 a:hover { color: var(--accent); }
.skill-card p { margin: 0; font-size: 12.5px; color: var(--muted); line-height: 1.6; }
.card-badges { display: flex; gap: 4px; flex-wrap: wrap; }
.card { transition: border-color var(--t-card), box-shadow var(--t-card), transform var(--t-card); }
.card:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); border-color: var(--line-strong); text-decoration: none; }
.wf-card { transition: border-color var(--t-card), box-shadow var(--t-card), transform var(--t-card); }
.wf-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); border-color: var(--line-strong); text-decoration: none; }

.sc-domain { font-size: 10.5px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin-bottom: 2px; flex-shrink: 0; }
.sc-title { font-size: 13.5px; font-weight: 650; line-height: 1.4; letter-spacing: -.02em; color: var(--ink); margin: 0 0 5px; flex-shrink: 0; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.sc-desc { font-size: 12.5px; color: var(--muted); line-height: 1.6; margin: 0 0 8px; flex: 1; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.sc-footer { display: flex; align-items: center; gap: 5px; margin-top: auto; flex-shrink: 0; min-height: 20px; }
.sc-roi { font-size: 10.5px; font-weight: 700; background: var(--green-bg); color: var(--green-dark); padding: 1px 7px; border-radius: var(--r-xs); white-space: nowrap; max-width: 130px; overflow: hidden; text-overflow: ellipsis; }
.sc-diff { font-size: 10.5px; color: var(--muted); padding: 1px 7px; background: var(--panel-2); border-radius: var(--r-xs); white-space: nowrap; }

.roi-badge, span.roi-badge {
  display: inline-flex; align-items: center;
  padding: 3px 9px;
  background: rgba(22,163,74,0.1);
  border: 1px solid rgba(22,163,74,0.25);
  border-radius: var(--r-full);
  font-size: 11px; font-weight: 700;
  color: var(--green-dark); letter-spacing: .01em;
}
.diff-badge { display: inline-block; padding: 2px 7px; border-radius: var(--r-xs); font-size: 10.5px; font-weight: 500; background: var(--panel-2); color: var(--muted); white-space: nowrap; }

.tag { display: inline-block; padding: 2px 7px; background: var(--panel-2); border: 1px solid var(--line); border-radius: var(--r-xs); font-size: 11px; color: var(--ink-2); text-decoration: none; font-weight: 500; white-space: nowrap; }
.tag:hover { background: var(--panel-3); border-color: var(--line-strong); text-decoration: none; }
.tag.topic { background: var(--green-bg); color: var(--green-dark); border-color: rgba(20,83,45,.12); }
.tag.topic:hover { background: #dcfce7; }
.tag-row { margin: 5px 0 12px; }

.breadcrumbs { color: var(--muted); margin-bottom: 12px; font-size: 12px; }
.breadcrumbs a { color: var(--accent); text-decoration: none; }
.breadcrumbs a:hover { text-decoration: underline; }
.two-col { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 24px; margin-top: 18px; }
.relation-panel { position: sticky; top: calc(var(--topbar-height) + 16px); align-self: start; background: var(--panel); border: 1px solid var(--line); border-radius: var(--r-lg); padding: 18px; }
.relation-panel h2 { margin: 0 0 10px; font-size: 13.5px; font-weight: 700; }
.relation-panel h3 { margin: 14px 0 5px; font-size: 10.5px; color: var(--muted); text-transform: uppercase; letter-spacing: .07em; font-weight: 700; }
.relation-panel ul { padding: 0; margin: 0; list-style: none; display: flex; flex-direction: column; }
.relation-panel ul li { border-bottom: 1px solid var(--line); }
.relation-panel ul li:last-child { border-bottom: none; }
.relation-panel ul li a { display: block; padding: 4px 0; font-size: 12px; color: var(--accent); text-decoration: none; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.relation-panel ul li a:hover { text-decoration: underline; }
#ego-graph { display: block; width: 100%; height: auto; min-height: 160px; border-radius: var(--r-md); background: var(--panel-2); border: 1px solid var(--line); margin-bottom: 8px; }
.ego-legend { font-size: 11px; color: var(--muted); display: flex; align-items: center; gap: 6px; margin-bottom: 12px; flex-wrap: wrap; }
.roi-panel { display: flex; gap: 16px; flex-wrap: wrap; background: var(--panel-2); border: 1px solid var(--line); border-radius: var(--r-md); padding: 12px 16px; margin: 10px 0 18px; }

/* ═══ Skill页面专属升级 ═══ */
.skill-header-block {
  margin-bottom: 20px;
  padding-bottom: 20px;
  border-bottom: 1px solid var(--line);
}
.skill-domain-chip {
  display: inline-flex; align-items: center;
  padding: 3px 10px;
  background: rgba(181,50,62,0.08);
  border: 1px solid rgba(181,50,62,0.18);
  border-radius: var(--r-full);
  font-size: 10.5px; font-weight: 700;
  letter-spacing: .08em; text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 10px;
}
.skill-main-title {
  font-size: clamp(20px, 2.5vw, 28px);
  font-weight: 800; letter-spacing: -.03em;
  line-height: 1.25; color: var(--ink);
  margin: 0 0 8px;
}
.skill-skill-id {
  font-size: 11.5px; color: var(--muted);
  font-family: "JetBrains Mono", monospace;
  letter-spacing: .02em; margin: 0;
}

/* ═══ 代码块升级 ═══ */
.code-wrap {
  position: relative; margin: 12px 0 20px;
  border-radius: var(--r-lg);
  overflow: hidden;
  box-shadow: var(--shadow-lg);
}
.code-header {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 16px;
  background: #252526;
  border-bottom: 1px solid #333;
}
.code-lang-badge {
  display: inline-flex; align-items: center;
  padding: 2px 8px;
  background: rgba(181,50,62,0.2);
  border: 1px solid rgba(181,50,62,0.3);
  border-radius: var(--r-xs);
  font-size: 10px; font-weight: 700;
  letter-spacing: .06em; text-transform: uppercase;
  color: #e07070;
  font-family: "JetBrains Mono", monospace;
}
.code-meta {
  font-size: 11px; color: #666;
  font-family: "JetBrains Mono", monospace;
  margin-left: auto;
}
.copy-btn {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 4px 12px;
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: var(--r-sm);
  color: #aaa; font-size: 11px; font-weight: 600;
  cursor: pointer; transition: all var(--t);
  font-family: var(--font);
}
.copy-btn:hover { background: rgba(255,255,255,0.14); color: #ddd; }
.copy-btn.copied { background: rgba(22,163,74,0.2); border-color: rgba(22,163,74,0.3); color: #4ade80; }
.code-preview {
  background: #1e1e1e !important;
  border: none;
  border-radius: 0;
  padding: 20px 24px;
  margin: 0;
  overflow-x: auto;
  font-family: "JetBrains Mono", "Fira Code", Menlo, monospace;
  font-size: 12.5px; line-height: 1.75;
  color: #d4d4d4;
  box-shadow: none;
}
.code-preview::-webkit-scrollbar { height: 4px; }
.code-preview::-webkit-scrollbar-track { background: #1e1e1e; }
.code-preview::-webkit-scrollbar-thumb { background: #444; border-radius: 2px; }

/* ═══ ROI Panel精致化 ═══ */
.roi-panel {
  display: flex; gap: 20px; flex-wrap: wrap;
  background: var(--panel);
  border: 1px solid var(--line);
  border-left: 3px solid var(--accent);
  border-radius: var(--r-lg);
  padding: 16px 20px;
  margin: 12px 0 20px;
  box-shadow: var(--shadow-sm);
}
.roi-panel > * {
  display: flex; flex-direction: column; gap: 2px;
}
.roi-panel strong {
  font-size: 13px; font-weight: 700; color: var(--ink);
}
.roi-panel span {
  font-size: 11px; color: var(--muted);
  text-transform: uppercase; letter-spacing: .05em;
}

.roi-item { display: flex; flex-direction: column; }
.roi-label { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; font-weight: 600; }
.roi-value { font-size: 15px; font-weight: 700; margin-top: 2px; color: var(--ink); }

.search-results { position: absolute; top: var(--topbar-height); left: 0; right: 0; z-index: 300; background: var(--panel); border: 1px solid var(--line); border-radius: 0 0 var(--r-xl) var(--r-xl); box-shadow: var(--shadow-lg); max-height: 480px; overflow-y: auto; margin: 0; }
.search-results.hidden { display: none; }
.search-results .result { display: block; padding: 10px 18px; text-decoration: none; color: var(--ink); border-bottom: 1px solid var(--line); font-size: 13px; transition: background var(--t); }
.search-results .result:hover { background: var(--panel-2); }
.search-results .result:last-child { border-bottom: none; }
.search-results .result strong { color: var(--accent); font-weight: 600; }

.wf-meta { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin: 10px 0; }
.wf-entry-question { font-size: 14.5px; font-weight: 600; }
.wf-outcomes { background: var(--green-bg); border: 1px solid #bbf7d0; border-radius: var(--r-lg); padding: 12px 16px; margin: 14px 0; }
.wf-outcomes h3 { margin: 0 0 6px; font-size: 12.5px; color: var(--green-dark); font-weight: 700; }
.wf-outcomes ul { margin: 0; padding-left: 16px; }
.wf-outcomes li { font-size: 13px; margin-bottom: 3px; }
.wf-tree { margin-top: 20px; }
.wf-step { background: var(--panel); border: 1px solid var(--line); border-radius: var(--r-lg); padding: 16px 18px; margin-bottom: 10px; border-left: 2px solid var(--accent); }
.wf-step-name { font-weight: 700; font-size: 14px; margin-bottom: 6px; }
.wf-question { font-size: 13.5px; margin: 6px 0; color: var(--accent); font-weight: 500; }
.wf-context { font-size: 12.5px; color: var(--muted); margin-bottom: 10px; line-height: 1.6; }
.wf-branches { display: flex; flex-direction: column; gap: 6px; margin-top: 8px; }
.wf-branch { border: 1px solid var(--line); border-radius: var(--r-md); overflow: hidden; }
.wf-branch > summary { padding: 9px 12px; cursor: pointer; font-size: 13px; font-weight: 500; background: var(--panel-2); list-style: none; position: relative; padding-left: 32px; transition: background var(--t); }
.wf-branch > summary:hover { background: var(--panel-3); }
.wf-branch > summary::before { content: ''; position: absolute; left: 12px; top: 50%; width: 5px; height: 5px; border-right: 1.5px solid var(--muted); border-bottom: 1.5px solid var(--muted); transform: translateY(-65%) rotate(-45deg); transition: transform var(--t); }
.wf-branch[open] > summary::before { transform: translateY(-35%) rotate(45deg); }
.wf-condition { color: var(--ink); }
.wf-branch-skills { display: flex; flex-wrap: wrap; gap: 6px; padding: 10px 12px; background: var(--panel); }
.wf-skill-chip { display: flex; flex-direction: column; background: var(--panel-2); border-radius: var(--r-md); padding: 7px 10px; text-decoration: none; color: var(--ink); min-width: 140px; border: 1px solid var(--line); transition: box-shadow var(--t), border-color var(--t); }
.wf-skill-chip:hover { box-shadow: var(--shadow-sm); border-color: var(--line-strong); }
.wf-skill-chip.missing { opacity: .4; cursor: default; }
.chip-name { font-size: 12px; font-weight: 700; color: var(--accent); }
.chip-role { font-size: 11.5px; color: var(--muted); margin-top: 1px; }

#graph-svg { width: 100%; display: block; background: var(--bg); border-radius: var(--r-lg); border: 1px solid var(--line); }
.graph-controls { display: flex; gap: 14px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; font-size: 13px; }
.graph-controls label { display: flex; align-items: center; gap: 5px; cursor: pointer; font-weight: 500; }
.edge-dot { width: 12px; height: 3px; border-radius: 2px; display: inline-block; }
.edge-dot.prereq { background: #B5323E; }
.edge-dot.combo  { background: #14532D; }
.edge-dot.ext    { background: #D97706; }
.graph-info { position: fixed; top: 70px; right: 20px; z-index: 20; background: var(--panel); border: 1px solid var(--line); border-radius: var(--r-xl); padding: 18px; width: 260px; box-shadow: var(--shadow-lg); }
.graph-info.hidden { display: none; }
.graph-info h3 { margin: 0 26px 8px 0; font-size: 13.5px; font-weight: 700; }
.graph-info p { margin: 3px 0; font-size: 12.5px; }

table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--line); }
th { background: var(--panel-2); font-weight: 700; font-size: 11.5px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
tr:hover td { background: var(--bg); }

.skill-toc { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 16px; }
.skill-toc a { padding: 3px 10px; border-radius: var(--r-xs); font-size: 11.5px; font-weight: 600; background: var(--panel-2); color: var(--ink-2); border: 1px solid var(--line); text-decoration: none; transition: background .12s, color .12s; white-space: nowrap; }
.skill-toc a:hover { background: var(--panel-3); color: var(--ink); border-color: var(--line-strong); }

.code-wrap { position: relative; margin: 10px 0; }
.copy-btn { position: absolute; top: 8px; right: 8px; z-index: 2; padding: 3px 9px; border-radius: var(--r-xs); border: 1px solid rgba(255,255,255,.14); background: rgba(255,255,255,.08); color: #d0c8c0; font-size: 11px; font-weight: 600; cursor: pointer; transition: background .12s; letter-spacing: .03em; }
.copy-btn:hover { background: rgba(255,255,255,.18); }
.copy-btn.copied { color: #6ee7b7; border-color: #6ee7b7; }
pre, .code-block {
  background: #1e1e1e !important;
  border: 1px solid #333;
  border-radius: var(--r-lg);
  padding: 20px 24px;
  overflow-x: auto;
  position: relative;
  font-family: "JetBrains Mono", "Fira Code", Menlo, monospace;
  font-size: 12.5px;
  line-height: 1.7;
  color: #d4d4d4;
  box-shadow: inset 0 1px 4px rgba(0,0,0,0.2), var(--shadow-md);
  margin: 0;
  white-space: pre;
}
pre::before {
  content: attr(data-lang);
  position: absolute; top: 8px; right: 12px;
  font-size: 10px; font-weight: 600;
  color: #555; letter-spacing: .08em;
  text-transform: uppercase;
}
code {
  background: rgba(0,0,0,0.06);
  border: 1px solid var(--line);
  border-radius: var(--r-xs);
  padding: 1px 5px;
  font-family: "JetBrains Mono", "Fira Code", Menlo, monospace;
  font-size: 12px;
  color: var(--accent);
}
pre code {
  background: none; border: none;
  padding: 0; color: inherit;
}
.code-preview {
  background: #1e1e1e !important;
  color: #d4d4d4;
  border-radius: var(--r-lg);
  padding: 20px 24px;
  overflow-x: auto; overflow-y: auto;
  font-size: 12.5px;
  line-height: 1.7;
  font-family: "JetBrains Mono", "Fira Code", Menlo, monospace;
  max-height: 400px; margin: 0; white-space: pre;
  border: 1px solid #333;
  box-shadow: inset 0 1px 4px rgba(0,0,0,0.2), var(--shadow-md);
}

.biz-ctx-panel { background: var(--panel); border: 1px solid var(--line); border-left: 3px solid var(--accent); border-radius: var(--r-lg); padding: 16px 20px; margin: 12px 0 18px; }
.biz-ctx-header { font-size: 10.5px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; color: var(--accent); margin-bottom: 12px; }
.biz-ctx-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 20px; }
.biz-ctx-item { display: flex; flex-direction: column; gap: 2px; }
.biz-ctx-full { grid-column: 1 / -1; }
.biz-ctx-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
.biz-ctx-value { font-size: 13px; color: var(--ink-2); line-height: 1.5; }
.biz-ctx-secondary { color: var(--muted); font-weight: 400; }
.biz-ctx-outcome { color: var(--green-dark); font-weight: 500; }
.biz-pain-tags { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 3px; }
.biz-pain-tag { font-size: 11.5px; padding: 2px 9px; background: var(--panel-2); border: 1px solid var(--line); border-radius: var(--r-xs); color: var(--ink-2); font-weight: 500; }
@media (max-width: 600px) { .biz-ctx-grid { grid-template-columns: 1fr; } }

.filter-bar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin: 14px 0 6px; }
.hb-uplinks { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; margin: 6px 0 10px; }
.hb-uplinks-label { font-size: 10.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
.hb-uplink { font-size: 11.5px; font-weight: 600; padding: 3px 10px; border-radius: var(--r-xs); background: var(--panel-2); color: var(--ink-2); text-decoration: none; border: 1px solid var(--line); transition: background var(--t), border-color var(--t); }
.hb-uplink:hover { background: var(--panel-3); border-color: var(--line-strong); }
.filter-select { padding: 6px 10px; border: 1px solid var(--line-strong); border-radius: var(--r-md); font-size: 13px; background: var(--panel); color: var(--ink); cursor: pointer; font-family: var(--font); transition: border-color var(--t); }
.filter-select:focus { outline: none; border-color: var(--accent); }
.filter-hint { font-size: 13px; color: var(--muted); }

.pb-hero { display: flex; gap: 14px; align-items: flex-start; margin-bottom: 12px; }
.pb-icon { width: 40px; height: 40px; flex-shrink: 0; display: flex; align-items: center; justify-content: center; background: var(--panel-3); color: var(--ink-2); border-radius: var(--r-lg); margin-top: 3px; border: 1px solid var(--line); overflow: hidden; }
.pb-icon svg { width: 20px; height: 20px; }
.pb-hero-body h1 { margin: 0 0 5px; }
.biz-tag { display: inline-block; font-size: 10.5px; font-weight: 600; background: var(--panel-2); color: var(--ink-2); padding: 2px 8px; border-radius: var(--r-xs); margin-bottom: 6px; border: 1px solid var(--line); }
.pb-roi-callout { display: inline-flex; align-items: center; gap: 10px; background: var(--red-bg); border: 1px solid rgba(220,38,38,.15); border-radius: var(--r-md); padding: 8px 14px; margin: 3px 6px 3px 0; font-size: 12.5px; font-weight: 500; }
.pb-roi-val { font-weight: 700; color: var(--red); font-size: 13.5px; }
.hero-badge { display: inline-block; font-size: 11px; font-weight: 600; letter-spacing: .07em; text-transform: uppercase; color: var(--accent); background: var(--accent-light); padding: 3px 10px; border-radius: var(--r-xs); margin: 0 0 10px; }
.hero-primary-cta { display: flex; gap: 8px; flex-wrap: wrap; margin: 0 0 20px; }
.rm-scqa { background: var(--panel-2); border: 1px solid var(--line); border-radius: var(--r-lg); padding: 20px 24px; margin: 0 0 28px; max-width: 860px; margin-left: auto; margin-right: auto; }
.rm-scqa-s, .rm-scqa-c, .rm-scqa-q { display: flex; gap: 10px; margin-bottom: 10px; font-size: 13.5px; line-height: 1.65; color: var(--ink-2); }
.rm-scqa-q { margin-bottom: 0; font-weight: 600; color: var(--ink); }
.rm-scqa-label { flex-shrink: 0; width: 32px; height: 18px; background: var(--accent); color: #fff; border-radius: var(--r-xs); font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; display: flex; align-items: center; justify-content: center; margin-top: 3px; }
.rm-scqa-c .rm-scqa-label { background: var(--red); }
.rm-scqa-q .rm-scqa-label { background: var(--amber-dark); }
.pb-intro { background: var(--panel-2); border-left: 2px solid var(--accent); border-radius: 0 var(--r-md) var(--r-md) 0; padding: 12px 16px; margin: 14px 0; font-size: 13.5px; color: var(--ink-2); line-height: 1.7; }
.pb-steps { margin-top: 20px; display: flex; flex-direction: column; gap: 12px; }
.pb-lead-capture { margin-top: 36px; padding: 28px; border-radius: var(--r-xl); background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); border: 1px solid #30363d; }
.pb-lead-inner { display: flex; gap: 28px; align-items: flex-start; flex-wrap: wrap; }
.pb-lead-text { flex: 1; min-width: 240px; }
.pb-lead-text h3 { color: #f0f6fc; font-size: 17px; margin: 0 0 8px; }
.pb-lead-text p { color: #8b949e; font-size: 13.5px; margin: 0 0 10px; }
.pb-lead-bullets { color: #8b949e; font-size: 12.5px; padding-left: 0; list-style: none; margin: 0; }
.pb-lead-bullets li { margin-bottom: 5px; }
.pb-lead-action { flex-shrink: 0; text-align: center; }
.pb-lead-btn { display: inline-block; padding: 12px 24px; background: var(--accent); color: #fff; border-radius: var(--r-md); font-weight: 700; font-size: 14px; text-decoration: none; transition: background .12s; }
.pb-lead-btn:hover { background: var(--accent-dark); }
.pb-lead-note { color: #484f58; font-size: 11.5px; margin-top: 8px; }
.pb-step { display: flex; gap: 16px; background: var(--panel); border: 1px solid var(--line); border-radius: var(--r-lg); padding: 18px; transition: border-color var(--t-card); }
.pb-step:hover { border-color: var(--line-strong); }
.pb-step-num { width: 28px; height: 28px; border-radius: 50%; flex-shrink: 0; background: var(--ink); color: #fff; display: flex; align-items: center; justify-content: center; font-size: 11px; font-weight: 700; margin-top: 3px; font-family: var(--font); }
.pb-step-body { flex: 1; min-width: 0; }
.pb-step-title { margin: 0 0 6px; font-size: 14.5px; font-weight: 600; letter-spacing: -.02em; }
.pb-problem { font-size: 12.5px; color: var(--accent); margin: 0 0 12px; font-weight: 500; padding: 7px 10px; background: var(--accent-light); border-radius: var(--r-xs); border-left: 2px solid var(--accent); }
.pb-skills { display: flex; flex-direction: column; gap: 6px; margin-bottom: 12px; }
.pb-skill { background: var(--panel-2); border: 1px solid var(--line); border-radius: var(--r-md); padding: 8px 12px; }
.pb-skill-header { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.pb-skill-name { font-weight: 700; font-size: 13px; color: var(--accent); text-decoration: none; }
.pb-skill-name:hover { text-decoration: underline; }
.pb-skill-badges { display: flex; gap: 5px; }
.pb-skill-why { margin: 4px 0 0; font-size: 12px; color: var(--muted); line-height: 1.5; }
.pb-data, .pb-output { font-size: 12px; margin-top: 8px; background: var(--panel-2); border: 1px solid var(--line); border-radius: var(--r-xs); padding: 6px 10px; }
.pb-outcomes { background: var(--green-bg); border: 1px solid rgba(22,163,74,.15); border-radius: var(--r-lg); padding: 16px 20px; margin-top: 20px; }
.pb-outcomes h2 { margin: 0 0 8px; font-size: 13.5px; color: var(--green-dark); font-weight: 700; }
.pb-outcomes ul { margin: 0; padding-left: 16px; }
.pb-outcomes li { font-size: 13px; margin-bottom: 4px; color: var(--green-dark); }

.calc-wrapper { margin: 36px 0 0; background: var(--panel); border: 1px solid var(--line); border-radius: var(--r-xl); overflow: hidden; }
.calc-header { padding: 20px 24px 16px; border-bottom: 1px solid var(--line); }
.calc-header h2 { margin: 0 0 4px; font-size: 19px; font-weight: 800; }
.calc-tabs { display: flex; gap: 0; border-bottom: 1px solid var(--line); background: var(--panel-2); }
.calc-tab { flex: 1; padding: 11px 8px; border: none; background: none; cursor: pointer; font-size: 12.5px; font-weight: 500; font-family: var(--font); color: var(--muted); border-bottom: 2px solid transparent; transition: color var(--t), background var(--t), border-color var(--t); }
.calc-tab:hover { color: var(--ink); background: var(--panel-3); }
.calc-tab.active { color: var(--tc, var(--accent)); border-bottom-color: var(--tc, var(--accent)); background: #fff; font-weight: 700; }
.calc-body { padding: 0; }
.calc-panel { display: none; grid-template-columns: 1fr 240px; gap: 0; }
.calc-panel.active { display: grid; }
.calc-inputs { padding: 20px 24px; display: flex; flex-direction: column; gap: 18px; }
.calc-row { display: flex; flex-direction: column; gap: 5px; }
.calc-label { font-size: 12.5px; font-weight: 600; color: var(--ink-2); }
.calc-input-wrap { display: flex; align-items: center; gap: 10px; }
.calc-input { flex: 1; accent-color: var(--tc, var(--accent)); height: 4px; cursor: pointer; }
.calc-val { font-size: 15px; font-weight: 800; color: var(--tc, var(--accent)); min-width: 52px; text-align: right; font-variant-numeric: tabular-nums; }
.calc-unit { font-size: 11.5px; color: var(--muted); min-width: 48px; }
.calc-result { background: var(--tc, var(--accent)); padding: 32px 20px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; }
.calc-result-label { font-size: 10.5px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: rgba(255,255,255,.7); margin-bottom: 8px; }
.calc-result-num { font-size: 48px; font-weight: 900; color: #fff; line-height: 1; font-variant-numeric: tabular-nums; }
.calc-result-unit { font-size: 14px; color: rgba(255,255,255,.85); margin-top: 5px; font-weight: 600; }
.calc-disclaimer { font-size: 10.5px; color: rgba(255,255,255,.5); margin-top: 16px; line-height: 1.6; max-width: 180px; }

.agent-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; margin: 12px 0; align-items: stretch; }
.agent-card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--r-xl);
  padding: 20px;
  display: flex; flex-direction: column; gap: 12px;
  transition: all var(--t-card);
  position: relative;
  cursor: pointer;
}
.agent-card:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-lg);
  border-color: rgba(181,50,62,0.15);
}
.agent-card-top { display: flex; align-items: flex-start; gap: 10px; }
.agent-icon-wrap {
  width: 44px; height: 44px;
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 18px; font-weight: 800;
  flex-shrink: 0;
  box-shadow: var(--shadow-sm);
  background: var(--panel-3); color: var(--ink-2);
  border: 1px solid var(--line);
}
.agent-icon-wrap.cat-supply { background: var(--amber-bg); color: var(--amber-dark); border-color: rgba(217,119,6,.2); }
.agent-icon-wrap.cat-ad { background: var(--accent-light); color: var(--accent); border-color: rgba(181,50,62,.2); }
.agent-icon-wrap.cat-risk { background: var(--red-bg); color: var(--red); border-color: rgba(220,38,38,.2); }
.agent-icon-wrap.cat-voc { background: var(--green-bg); color: var(--green-dark); border-color: rgba(22,163,74,.2); }
.agent-icon-wrap.cat-ops { background: #f0f4ff; color: #3b5bdb; border-color: rgba(59,91,219,.15); }
.agent-icon-wrap.cat-finance { background: #f0fdf4; color: #166534; border-color: rgba(22,163,74,.2); }
.agent-icon-wrap.cat-compliance { background: #fff7ed; color: #9a3412; border-color: rgba(234,88,12,.2); }
.agent-icon-wrap svg { width: 22px; height: 22px; }
.agent-card-info { flex: 1; min-width: 0; }
.agent-name { font-size: 13.5px; font-weight: 700; margin: 0 0 3px; letter-spacing: -.02em; }
.agent-cat-badge { display: inline-block; font-size: 10px; font-weight: 600; padding: 1px 6px; border-radius: var(--r-xs); background: var(--panel-2); color: var(--muted); border: 1px solid var(--line); }
.agent-status { display: flex; align-items: center; gap: 5px; font-size: 11.5px; }
.status-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--green); animation: pulse-dot 2.5s ease-in-out infinite; }
.status-dot.demo { background: var(--amber); }
.status-dot.live { background: #16A34A; box-shadow: 0 0 0 2px rgba(22,163,74,.15); }
@keyframes pulse-dot { 0%,100%{transform:scale(1);opacity:1} 50%{transform:scale(1.3);opacity:.7} }
.agent-desc { font-size: 12px; color: var(--muted); line-height: 1.55; margin: 0; }
.agent-roi { font-size: 10.5px; font-weight: 600; color: var(--green-dark); background: var(--green-bg); padding: 2px 7px; border-radius: var(--r-xs); align-self: flex-start; }
.agent-skills { display: flex; flex-wrap: wrap; gap: 4px; }
.agent-skill-chip { font-size: 10px; background: var(--panel-2); color: var(--ink-2); padding: 1px 6px; border-radius: var(--r-xs); font-weight: 500; text-decoration: none; border: 1px solid var(--line); }
.agent-invoke-btn { margin-top: auto; width: 100%; padding: 8px; background: var(--ink); color: #fff; border: none; border-radius: var(--r-md); font-size: 12.5px; font-weight: 600; font-family: var(--font); letter-spacing: -.01em; cursor: pointer; transition: background var(--t), box-shadow var(--t); display: flex; align-items: center; justify-content: center; gap: 6px; }
.agent-invoke-btn:hover { background: var(--ink-2); box-shadow: var(--shadow-md); }
.agent-cat-filter { display: flex; gap: 5px; flex-wrap: wrap; margin: 12px 0 5px; }
.cat-pill { padding: 5px 12px; border-radius: var(--r-sm); border: 1px solid var(--line-strong); background: var(--panel); font-size: 11.5px; font-weight: 500; font-family: var(--font); color: var(--muted); cursor: pointer; transition: all var(--t); }
.cat-pill:hover { border-color: var(--ink); color: var(--ink); }
.cat-pill.active { background: var(--ink); border-color: var(--ink); color: #fff; }

.agent-modal-overlay { position: fixed; inset: 0; z-index: 1000; background: rgba(0,0,0,.6); backdrop-filter: blur(4px); display: flex; align-items: center; justify-content: center; padding: 20px; opacity: 0; pointer-events: none; transition: opacity var(--t-slow); }
.agent-modal-overlay.open { opacity: 1; pointer-events: all; }
.agent-modal { background: var(--panel); border-radius: var(--r-xl); width: 100%; max-width: 640px; max-height: 88vh; overflow-y: auto; box-shadow: 0 20px 40px rgba(0,0,0,.15); transform: translateY(12px) scale(.98); transition: transform var(--t-slow); }
.agent-modal-overlay.open .agent-modal { transform: translateY(0) scale(1); }
.modal-header { position: sticky; top: 0; z-index: 1; display: flex; align-items: center; gap: 12px; padding: 16px 20px; background: var(--panel); border-bottom: 1px solid var(--line); }
.modal-icon { width: 44px; height: 44px; border-radius: var(--r-lg); background: var(--panel-3); color: var(--ink-2); display: flex; align-items: center; justify-content: center; flex-shrink: 0; border: 1px solid var(--line); overflow: hidden; }
.modal-icon svg { width: 22px; height: 22px; }
.modal-header-info { flex: 1; }
.modal-header-info h2 { margin: 0 0 3px; font-size: 15px; font-weight: 700; }
.modal-close { width: 28px; height: 28px; border-radius: 50%; background: var(--panel-2); border: 1px solid var(--line); cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 13px; transition: background var(--t); color: var(--muted); font-weight: 500; }
.modal-close:hover { background: var(--red-bg); color: var(--red); }
.modal-body { padding: 20px; }
.modal-section { margin-bottom: 20px; }
.modal-section h3 { font-size: 10.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .07em; color: var(--muted); margin: 0 0 10px; }
.modal-input-group { display: flex; flex-direction: column; gap: 8px; }
.modal-input { width: 100%; padding: 8px 12px; border: 1px solid var(--line-strong); border-radius: var(--r-md); font-size: 13px; background: var(--panel); color: var(--ink); font-family: var(--font); transition: border-color var(--t); }
.modal-input:focus { outline: none; border-color: var(--accent); }
.modal-input::placeholder { color: var(--muted); }
.modal-run-btn { width: 100%; padding: 11px; background: var(--ink); color: #fff; border: none; border-radius: var(--r-md); font-size: 13px; font-weight: 700; font-family: var(--font); letter-spacing: -.01em; cursor: pointer; transition: background var(--t), box-shadow var(--t); display: flex; align-items: center; justify-content: center; gap: 8px; }
.modal-run-btn:hover { background: var(--ink-2); box-shadow: var(--shadow-md); }
.modal-run-btn:disabled { background: var(--line-strong); cursor: not-allowed; }
.modal-output { background: var(--panel-2); border: 1px solid var(--line); border-radius: var(--r-md); padding: 14px 16px; margin-top: 12px; min-height: 90px; display: none; }
.modal-output.visible { display: block; }
.output-thinking { display: flex; align-items: center; gap: 10px; color: var(--muted); font-size: 13px; }
.thinking-dots span { animation: blink 1.2s infinite; font-size: 16px; letter-spacing: 2px; }
.thinking-dots span:nth-child(2) { animation-delay: .2s; }
.thinking-dots span:nth-child(3) { animation-delay: .4s; }
@keyframes blink { 0%,100%{opacity:.2} 50%{opacity:1} }
.output-content { font-size: 13px; line-height: 1.7; white-space: pre-wrap; word-break: break-word; font-family: var(--font); }
.modal-footer-skills { display: flex; flex-wrap: wrap; gap: 5px; margin-top: 12px; }

.agent-hero { display: flex; justify-content: space-between; align-items: flex-start; gap: 20px; margin-bottom: 22px; flex-wrap: wrap; }
.agent-hero-text { flex: 1; min-width: 260px; }
.agent-hero-stats { display: flex; gap: 10px; flex-shrink: 0; }
.agent-stat { background: var(--panel); border: 1px solid var(--line); border-radius: var(--r-lg); padding: 12px 16px; text-align: center; min-width: 70px; }
.agent-stat strong { display: block; font-size: 32px; font-weight: 800; color: var(--ink); letter-spacing: -.05em; }
.agent-stat span { font-size: 10.5px; color: var(--muted); font-weight: 500; }
@media(max-width: 600px) { .agent-hero { flex-direction: column; } .agent-hero-stats { width: 100%; justify-content: space-around; } }

.sd-hero {
  padding: 40px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--r-xl);
  display: flex; gap: 28px; align-items: flex-start;
  margin-bottom: 28px;
  box-shadow: var(--shadow-sm);
  position: relative; overflow: hidden;
}
.sd-hero::after {
  content: '';
  position: absolute; top: 0; right: 0;
  width: 300px; height: 100%;
  background: var(--grad-glow);
  pointer-events: none;
}
.sd-hero-icon {
  width: 64px; height: 64px; flex-shrink: 0;
  border-radius: 14px;
  display: flex; align-items: center; justify-content: center;
  font-size: 24px; font-weight: 800;
  box-shadow: var(--shadow-md);
  background: var(--panel-3); color: var(--ink-2);
  border: 1px solid var(--line);
}
.rm-hero{text-align:center;padding:50px 32px 36px;max-width:800px;margin:0 auto}
.rm-hero-eyebrow{font-size:12px;font-weight:600;letter-spacing:.08em;color:var(--muted);text-transform:uppercase;margin-bottom:14px}
.rm-hero-title{font-size:34px;font-weight:800;line-height:1.1;margin:0 0 14px;color:var(--ink);letter-spacing:-.04em}
.rm-hero-sub{font-size:17px;color:var(--muted);margin:0 0 24px;line-height:1.5}
.rm-hero-cta{display:flex;gap:10px;justify-content:center;margin-bottom:12px}
.rm-hero-note{font-size:11.5px;color:var(--muted)}
.rm-btn-primary{padding:11px 24px;background:var(--ink);color:#fff;border:none;border-radius:var(--r-md);font-size:14px;font-weight:600;cursor:pointer;text-decoration:none;display:inline-block}
.rm-btn-primary:hover{background:var(--ink-2)}
.rm-btn-sec{padding:11px 24px;background:var(--panel-2);color:var(--ink);border:1px solid var(--line-strong);border-radius:var(--r-md);font-size:14px;font-weight:600;text-decoration:none;display:inline-block}

.rm-summary-bar{display:flex;align-items:center;justify-content:center;gap:0;background:var(--ink);color:#fff;padding:20px 32px;margin:0 -40px;flex-wrap:wrap;gap:6px}
.rm-summary-item{text-align:center;padding:0 20px}
.rm-summary-num{display:block;font-size:26px;font-weight:800;color:#EEEEEE;letter-spacing:-.04em}
.rm-summary-label{font-size:11.5px;color:#888888}
.rm-summary-sep{font-size:20px;color:#333333;padding:0 6px}

.rm-roles-bar{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:28px 0;padding:0}
.rm-role{display:flex;gap:14px;align-items:flex-start;background:var(--panel);border:1px solid var(--line);border-radius:var(--r-lg);padding:16px}
.rm-role-icon{font-size:28px;flex-shrink:0}
.rm-role strong{display:block;font-size:14px;margin-bottom:3px}
.rm-role p{margin:0 0 6px;font-size:12.5px;color:var(--muted)}
.rm-role-roi{font-size:12.5px;font-weight:700;color:var(--accent);background:var(--accent-light);padding:2px 8px;border-radius:var(--r-xs)}

.rm-phases{display:flex;flex-direction:column;gap:20px;margin:28px 0}
.rm-phase{border-radius:var(--r-lg);overflow:hidden;border:1px solid var(--line)}
.rm-phase-header{display:flex;align-items:flex-start;gap:18px;padding:20px 24px;background:var(--phase-bg);border-bottom:1px solid var(--line)}
.rm-phase-badge{width:64px;height:64px;border-radius:50%;background:var(--phase-color);color:#fff;display:flex;align-items:center;justify-content:center;font-weight:800;font-size:12px;flex-shrink:0;text-align:center;line-height:1.2}
.rm-phase-meta{flex:1}
.rm-phase-period{font-size:11px;font-weight:600;color:var(--phase-color);text-transform:uppercase;letter-spacing:.06em}
.rm-phase-theme{margin:3px 0;font-size:20px;font-weight:800;letter-spacing:-.03em}
.rm-phase-tagline{margin:0;font-size:13.5px;color:var(--muted)}
.rm-phase-roi{text-align:right;flex-shrink:0}
.rm-phase-roi-label{display:block;font-size:10.5px;color:var(--muted);margin-bottom:3px;text-transform:uppercase;letter-spacing:.04em}
.rm-phase-roi strong{font-size:17px;font-weight:800;color:var(--phase-color)}

.rm-items{display:flex;flex-direction:column;gap:0}
.rm-item{display:flex;gap:18px;padding:20px 24px;border-bottom:1px solid var(--line)}
.rm-item:last-child{border-bottom:none}
.rm-item-icon{font-size:24px;flex-shrink:0;width:36px;text-align:center;margin-top:2px}
.rm-item-body{flex:1}
.rm-item-title{margin:0 0 10px;font-size:15px;font-weight:700;letter-spacing:-.02em}
.rm-story{background:var(--amber-bg);border-left:2px solid var(--amber);padding:8px 12px;border-radius:0 var(--r-sm) var(--r-sm) 0;font-size:12.5px;color:var(--amber-dark);margin-bottom:6px}
.rm-story-label,.rm-result-label{font-weight:700;margin-right:5px}
.rm-result{background:var(--green-bg);border-left:2px solid var(--green);padding:8px 12px;border-radius:0 var(--r-sm) var(--r-sm) 0;font-size:12.5px;color:var(--green-dark);margin-bottom:6px}
.rm-roi-line{font-size:12.5px;margin-bottom:6px;color:var(--ink-2)}
.rm-chips{display:flex;flex-wrap:wrap;gap:5px}
.rm-chip{font-size:10.5px;background:var(--accent-light);color:var(--accent);padding:2px 8px;border-radius:var(--r-xs);text-decoration:none;border:1px solid rgba(181,50,62,.15)}
.rm-chip:hover{background:var(--accent-bg)}

.rm-footer{display:grid;grid-template-columns:1fr 300px;gap:36px;margin-top:36px;padding:28px 32px;background:var(--ink);border-radius:var(--r-xl);color:#CCCCCC}
.rm-footer h3{margin:0 0 8px;font-size:17px;color:#EEEEEE}
.rm-footer p{font-size:13.5px;color:#888888;margin:0 0 14px}
.rm-footer-links{display:flex;flex-direction:column;gap:6px}
.rm-footer-links a{color:#B5323E;text-decoration:none;font-size:13.5px}
.rm-footer-links a:hover{text-decoration:underline}
.rm-footer-right{display:flex;flex-direction:column;justify-content:space-between}
.rm-footer-cta{text-align:center;background:#1a1a1a;border-radius:var(--r-lg);padding:20px;margin-bottom:14px}
.rm-footer-cta p{color:#888888;font-size:12.5px;margin-bottom:10px}
.rm-footer-note{font-size:11px;color:#444444;line-height:1.6}

@media (max-width: 1024px) { .layout { grid-template-columns: 190px 1fr; } .content { padding: 20px 24px; } .two-col { grid-template-columns: 1fr 290px; } }
@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; }
  .hamburger { display: flex; }
  .sidebar { display: none; position: fixed; top: var(--topbar-height); left: 0; width: 220px; height: calc(100vh - var(--topbar-height)); z-index: 195; transform: translateX(-100%); transition: transform var(--t-slow); box-shadow: var(--shadow-lg); }
  .sidebar.open { display: flex; flex-direction: column; transform: translateX(0); }
  #global-search { width: 140px; }
  .content { padding: 16px 14px; }
  .rm-summary-bar { margin: 0; }
}
@media (max-width: 768px) { .two-col { grid-template-columns: 1fr; } .relation-panel { position: static; } .ceo-entry { grid-template-columns: 1fr; } .biz-grid { grid-template-columns: 1fr; } .hero-tabs { flex-wrap: wrap; } .calc-panel.active { grid-template-columns: 1fr; } .calc-result { padding: 20px; } .calc-result-num { font-size: 38px; } .agent-grid { grid-template-columns: 1fr; } .agent-modal { max-height: 96vh; } .rm-roles-bar { grid-template-columns: 1fr; } .rm-footer { grid-template-columns: 1fr; } .rm-phase-header { flex-wrap: wrap; } }
@media (max-width: 480px) { .content h1 { font-size: 24px; } .pb-step { flex-direction: column; } .topbar { padding: 0 12px; } #global-search { width: 80px; font-size: 12px; } .metrics { grid-template-columns: repeat(2, 1fr); } .ds-grid { grid-template-columns: 1fr; } .biz-grid { grid-template-columns: 1fr; } }

.skill-toc { display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 14px; }
.skill-toc a { padding: 3px 9px; border-radius: var(--r-xs); font-size: 11px; font-weight: 600; background: var(--panel-2); color: var(--ink-2); border: 1px solid var(--line); text-decoration: none; transition: background .12s, color .12s; white-space: nowrap; }
.skill-toc a:hover { background: var(--panel-3); color: var(--ink); border-color: var(--line-strong); }

.biz-ctx-panel { background: var(--panel); border: 1px solid var(--line); border-left: 2px solid var(--accent); border-radius: var(--r-lg); padding: 14px 18px; margin: 10px 0 16px; }
.biz-ctx-header { font-size: 10px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; color: var(--accent); margin-bottom: 10px; }
.biz-ctx-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 18px; }
.biz-ctx-item { display: flex; flex-direction: column; gap: 2px; }
.biz-ctx-full { grid-column: 1 / -1; }
.biz-ctx-label { font-size: 9.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
.biz-ctx-value { font-size: 12.5px; color: var(--ink-2); line-height: 1.5; }
.biz-ctx-secondary { color: var(--muted); font-weight: 400; }
.biz-ctx-outcome { color: var(--green-dark); font-weight: 500; }
.biz-pain-tags { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 3px; }
.biz-pain-tag { font-size: 11px; padding: 2px 8px; background: var(--panel-2); border: 1px solid var(--line); border-radius: var(--r-xs); color: var(--ink-2); font-weight: 500; }
@media (max-width: 600px) { .biz-ctx-grid { grid-template-columns: 1fr; } }

.filter-bar { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin: 12px 0 5px; }
.hb-uplinks { display: flex; align-items: center; gap: 5px; flex-wrap: wrap; margin: 5px 0 8px; }
.hb-uplinks-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
.hb-uplink { font-size: 11px; font-weight: 600; padding: 2px 9px; border-radius: var(--r-xs); background: var(--panel-2); color: var(--ink-2); text-decoration: none; border: 1px solid var(--line); transition: background var(--t); }
.hb-uplink:hover { background: var(--panel-3); }
.filter-select { padding: 6px 10px; border: 1px solid var(--line-strong); border-radius: var(--r-md); font-size: 12.5px; background: var(--panel); color: var(--ink); cursor: pointer; font-family: var(--font); }
.filter-hint { font-size: 12.5px; color: var(--muted); }

@media print { .topbar,.sidebar,.rm-hero-cta,.rm-footer-cta button{display:none!important} body{background:#fff} .content{padding:0!important;max-width:100%!important} .rm-summary-bar{margin:0!important;-webkit-print-color-adjust:exact;print-color-adjust:exact} .rm-phase,.rm-footer{break-inside:avoid} .rm-phases{gap:12px} @page{margin:20mm 15mm;size:A4} }


/* ═══ Hot Skills 列表精致化 ═══ */
.hot-skill-link {
  display: flex; align-items: baseline; gap: 8px;
  color: var(--ink); text-decoration: none;
  font-size: 13px; font-weight: 500;
  transition: color var(--t);
}
.hot-skill-link:hover { color: var(--accent); text-decoration: none; }
.hot-skill-domain {
  font-size: 9.5px; font-weight: 700;
  text-transform: uppercase; letter-spacing: .08em;
  color: var(--muted); flex-shrink: 0;
  min-width: 80px;
}

/* ═══ Tabs 精致化 ═══ */
.hero-tabs {
  display: flex; gap: 4px; flex-wrap: wrap;
  justify-content: center;
  margin: 32px 0 16px;
  padding: 4px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--r-lg);
  display: inline-flex;
}
.tab-btn {
  padding: 7px 16px;
  border: none; border-radius: var(--r-md);
  background: transparent;
  color: var(--muted); font-size: 13px; font-weight: 500;
  cursor: pointer; transition: all var(--t);
  font-family: var(--font);
  white-space: nowrap;
}
.tab-btn:hover { color: var(--ink); background: var(--panel-2); }
.tab-btn.active {
  background: var(--ink);
  color: #fff; font-weight: 600;
  box-shadow: var(--shadow-sm);
}

/* ═══ Agent卡片分类色系升级 ═══ */
.cat-supply .agent-icon-wrap { background: rgba(59,130,246,0.1); color: #3b82f6; border: 1.5px solid rgba(59,130,246,0.2); }
.cat-ad .agent-icon-wrap     { background: rgba(181,50,62,0.1);  color: var(--accent); border: 1.5px solid rgba(181,50,62,0.2); }
.cat-risk .agent-icon-wrap   { background: rgba(234,88,12,0.1);  color: #ea580c; border: 1.5px solid rgba(234,88,12,0.2); }
.cat-voc .agent-icon-wrap    { background: rgba(22,163,74,0.1);  color: var(--green); border: 1.5px solid rgba(22,163,74,0.2); }
.cat-selection .agent-icon-wrap { background: rgba(139,92,246,0.1); color: #8b5cf6; border: 1.5px solid rgba(139,92,246,0.2); }
.agent-name {
  font-size: 14px; font-weight: 700; color: var(--ink);
  letter-spacing: -.02em;
}
.agent-category {
  font-size: 10px; font-weight: 700;
  text-transform: uppercase; letter-spacing: .08em;
  color: var(--muted); margin-bottom: 4px;
}
.agent-desc {
  font-size: 12.5px; color: var(--ink-2);
  line-height: 1.55; flex: 1;
}
.agent-roi {
  display: inline-flex; align-items: center;
  padding: 3px 8px;
  background: var(--green-bg);
  border: 1px solid rgba(22,163,74,0.2);
  border-radius: var(--r-full);
  font-size: 10.5px; font-weight: 700;
  color: var(--green-dark);
}

/* ═══ Playbook步骤精致化 ═══ */
.pb-step {
  border: 1px solid var(--line);
  border-radius: var(--r-lg);
  margin-bottom: 12px;
  overflow: hidden;
  transition: box-shadow var(--t-card);
}
.pb-step:hover { box-shadow: var(--shadow-md); }
.pb-step-header {
  padding: 14px 18px;
  background: var(--panel);
  display: flex; align-items: center; gap: 12px;
  cursor: pointer;
  border-bottom: 1px solid var(--line);
}
.pb-step-num {
  width: 28px; height: 28px; flex-shrink: 0;
  border-radius: 50%;
  background: var(--grad-accent);
  color: #fff; font-size: 13px; font-weight: 800;
  display: flex; align-items: center; justify-content: center;
  box-shadow: var(--shadow-accent);
}
.pb-step-title { font-size: 14px; font-weight: 700; color: var(--ink); flex: 1; }

/* ═══ Solution层次感 ═══ */
.sd-phase {
  padding: 12px 16px;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--r-md);
  margin-bottom: 8px;
  display: flex; gap: 14px; align-items: flex-start;
  transition: all var(--t-card);
}
.sd-phase:hover { border-color: rgba(181,50,62,0.2); box-shadow: var(--shadow-sm); }
.sd-phase-num {
  width: 32px; height: 32px; flex-shrink: 0;
  border-radius: var(--r-md);
  background: var(--accent-light);
  color: var(--accent); font-weight: 800; font-size: 14px;
  display: flex; align-items: center; justify-content: center;
}

/* ═══ Workflow步骤树精致化 ═══ */
.wf-step-block {
  border: 1px solid var(--line);
  border-radius: var(--r-lg);
  margin-bottom: 10px;
  overflow: hidden;
}
.wf-step-q {
  padding: 12px 18px;
  background: var(--panel);
  font-size: 14px; font-weight: 700;
  color: var(--ink);
  border-bottom: 1px solid var(--line);
  display: flex; align-items: center; gap: 8px;
}
.wf-step-q::before {
  content: '?';
  width: 22px; height: 22px;
  background: var(--ink); color: #fff;
  border-radius: 50%; font-size: 12px; font-weight: 800;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}

/* ═══ Diagnostic页面精致化 ═══ */
.diag-wrap {
  display: grid; grid-template-columns: 320px 1fr;
  gap: 20px; min-height: 600px;
  margin-top: 20px;
}
.diag-left {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--r-xl);
  padding: 20px;
  box-shadow: var(--shadow-sm);
  height: fit-content;
  position: sticky; top: calc(var(--topbar-height) + 20px);
}
.diag-label {
  font-size: 10.5px; font-weight: 700;
  text-transform: uppercase; letter-spacing: .08em;
  color: var(--muted); margin-bottom: 8px;
}
.diag-search-wrap {
  position: relative; margin-bottom: 16px;
}
.diag-search-wrap input {
  width: 100%; padding: 10px 14px 10px 36px;
  border: 1.5px solid var(--line-strong);
  border-radius: var(--r-lg);
  font-size: 13px; background: var(--panel);
  color: var(--ink); font-family: var(--font);
  transition: all var(--t);
  box-shadow: var(--shadow-inner);
}
.diag-search-wrap input:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(181,50,62,0.1);
}

/* ═══ 通用精致化工具类 ═══ */
.text-gradient {
  background: var(--grad-accent);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.glass-card {
  background: rgba(255,255,255,0.7);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(255,255,255,0.5);
}
.accent-line {
  border-left: 3px solid var(--accent);
  padding-left: 12px;
}
.number-display {
  font-family: var(--font-num, var(--font));
  font-size: 28px; font-weight: 800;
  letter-spacing: -.06em; line-height: 1;
  color: var(--ink);
}
.number-display.accent { color: var(--accent); }
.shimmer-bg {
  background: linear-gradient(
    90deg,
    var(--panel-2) 25%,
    var(--panel) 50%,
    var(--panel-2) 75%
  );
  background-size: 200% 100%;
  animation: shimmer 1.5s infinite;
}


/* ═══════════════════════════════════════════════════════
   GALLERY GRID — 5列大卡片浏览系统
   用于: 按领域浏览 / 按主题浏览 / 业务工作流
   ═══════════════════════════════════════════════════════ */

.gallery-grid {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 16px;
  margin: 24px 0;
}

@media (max-width: 1400px) { .gallery-grid { grid-template-columns: repeat(4, 1fr); } }
@media (max-width: 1100px) { .gallery-grid { grid-template-columns: repeat(3, 1fr); } }
@media (max-width: 768px)  { .gallery-grid { grid-template-columns: repeat(2, 1fr); } }
@media (max-width: 480px)  { .gallery-grid { grid-template-columns: 1fr; } }

/* ── 卡片基座 ── */
.gallery-card {
  position: relative;
  display: flex;
  flex-direction: column;
  height: 220px;
  border-radius: 14px;
  overflow: hidden;
  text-decoration: none;
  color: var(--ink);
  background: var(--panel);
  border: 1px solid var(--line);
  transition: transform .22s cubic-bezier(0.34, 1.56, 0.64, 1),
              box-shadow .22s ease,
              border-color .15s ease;
  cursor: pointer;
  isolation: isolate;
}

/* 悬浮效果 */
.gallery-card:hover {
  transform: translateY(-6px) scale(1.015);
  box-shadow: 0 20px 40px rgba(0,0,0,.12), 0 6px 12px rgba(0,0,0,.06);
  border-color: rgba(0,0,0,.08);
  text-decoration: none;
}

/* ── 背景图层 ── */
.gallery-card-bg {
  position: absolute;
  inset: 0;
  background-color: var(--card-bg, #f5f5f5);
  background-size: cover;
  background-repeat: repeat;
  background-position: center;
  opacity: 0.6;
  transition: opacity .22s ease, transform .35s ease;
  z-index: 0;
}
.gallery-card:hover .gallery-card-bg {
  opacity: 0.4;
  transform: scale(1.05);
}

/* 彩色渐变遮罩（品牌色） */
.gallery-card::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(
    145deg,
    color-mix(in srgb, var(--card-color, #555) 8%, transparent) 0%,
    color-mix(in srgb, var(--card-color, #555) 3%, transparent) 60%,
    transparent 100%
  );
  z-index: 1;
  transition: opacity .22s ease;
}
.gallery-card:hover::before {
  opacity: 0.7;
}

/* 底部渐变蒙层（确保footer可读） */
.gallery-card::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 80px;
  background: linear-gradient(
    to top,
    rgba(255,255,255,0.92) 0%,
    rgba(255,255,255,0.6) 50%,
    transparent 100%
  );
  z-index: 2;
}

/* ── 卡片正文 ── */
.gallery-card-body {
  position: relative;
  z-index: 3;
  flex: 1;
  padding: 18px 18px 8px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

/* 编号徽章 */
.gallery-card-num {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 28px; height: 28px;
  background: var(--card-color, #555);
  color: #fff;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 800;
  letter-spacing: -.01em;
  font-family: "JetBrains Mono", monospace;
  flex-shrink: 0;
  box-shadow: 0 2px 6px rgba(0,0,0,.15);
}
.gallery-card-num-sm {
  font-size: 9px;
  width: 32px;
  height: 20px;
  border-radius: 4px;
  letter-spacing: .02em;
}

/* 图标 */
.gallery-card-icon {
  font-size: 28px;
  line-height: 1;
  filter: drop-shadow(0 2px 4px rgba(0,0,0,.1));
  margin-top: 4px;
}
.gallery-card-icon-lg { font-size: 36px; margin-top: 8px; }

/* 标题 */
.gallery-card-title {
  font-size: 13.5px;
  font-weight: 700;
  color: var(--ink);
  letter-spacing: -.02em;
  line-height: 1.3;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  margin-top: auto;
}

/* 描述 */
.gallery-card-desc {
  font-size: 11px;
  color: var(--muted);
  line-height: 1.4;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* ── Footer（数量徽章）── */
.gallery-card-footer {
  position: relative;
  z-index: 3;
  padding: 10px 18px 14px;
  display: flex;
  align-items: baseline;
  gap: 4px;
  background: transparent;
}

.gallery-card-count {
  font-size: 22px;
  font-weight: 800;
  color: var(--card-color, #555);
  font-family: "JetBrains Mono", var(--font-num, var(--font));
  letter-spacing: -.04em;
  line-height: 1;
  transition: color .15s;
}
.gallery-card-unit {
  font-size: 11px;
  font-weight: 600;
  color: var(--muted);
  letter-spacing: .02em;
  text-transform: uppercase;
}

/* ── 领域卡片专属 ── */
.domain-gallery-card .gallery-card-num {
  font-size: 13px;
  width: 36px; height: 36px;
  border-radius: 8px;
}
.domain-gallery-card .gallery-card-title {
  font-size: 14px;
}
.domain-gallery-card .gallery-card-desc {
  font-size: 11px;
  color: var(--ink-2);
  font-weight: 500;
}

/* ── 主题卡片专属 ── */
.topic-gallery-card {
  height: 200px;
}
.topic-gallery-card .gallery-card-title {
  font-size: 13px;
  font-weight: 700;
}

/* ── 工作流卡片专属 ── */
.wf-gallery-card {
  height: 210px;
}
.wf-gallery-card .gallery-card-footer {
  padding-top: 8px;
}

/* ── 卡片入场动画 ── */
.gallery-card {
  animation: fadeInUp .35s ease both;
}
.gallery-grid .gallery-card:nth-child(1)  { animation-delay: .00s; }
.gallery-grid .gallery-card:nth-child(2)  { animation-delay: .03s; }
.gallery-grid .gallery-card:nth-child(3)  { animation-delay: .06s; }
.gallery-grid .gallery-card:nth-child(4)  { animation-delay: .09s; }
.gallery-grid .gallery-card:nth-child(5)  { animation-delay: .12s; }
.gallery-grid .gallery-card:nth-child(6)  { animation-delay: .06s; }
.gallery-grid .gallery-card:nth-child(7)  { animation-delay: .09s; }
.gallery-grid .gallery-card:nth-child(8)  { animation-delay: .12s; }
.gallery-grid .gallery-card:nth-child(9)  { animation-delay: .15s; }
.gallery-grid .gallery-card:nth-child(10) { animation-delay: .12s; }

/* ── color-mix降级兼容 ── */
@supports not (color: color-mix(in srgb, red 50%, blue)) {
  .gallery-card::before {
    background: linear-gradient(145deg, rgba(0,0,0,.04) 0%, transparent 100%);
  }
}

"""


