import { chromium } from 'playwright';
import { spawn } from 'node:child_process';
import path from 'node:path';
import fs from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');
const repoRoot = path.resolve(projectRoot, '../../../..');
const distDir = path.join(projectRoot, 'dist');
const relativeIndex = path.relative(repoRoot, path.join(projectRoot, 'index.html'));
const indexUrlPath = '/' + relativeIndex.split(path.sep).join('/');

const PORT = 8766;
const baseUrl = `http://127.0.0.1:${PORT}`;

const flavors = [
  { mode: 'main',  file: 'nlp-voc-report.pdf',            label: '主版 · 不含 Speaker Notes' },
  { mode: 'notes', file: 'nlp-voc-report-with-notes.pdf', label: '附录版 · 含 Speaker Notes' },
];

const log = (...args) => console.log('[export-pdf]', ...args);

async function startServer() {
  const proc = spawn('python3', ['-m', 'http.server', String(PORT), '--bind', '127.0.0.1'], {
    cwd: repoRoot,
    stdio: 'ignore',
    detached: true,
  });
  proc.unref();
  for (let i = 0; i < 30; i++) {
    try {
      const res = await fetch(`${baseUrl}${indexUrlPath}`);
      if (res.ok) {
        log(`server up @ ${baseUrl} · serving ${repoRoot}`);
        return proc;
      }
    } catch {}
    await new Promise(r => setTimeout(r, 200));
  }
  throw new Error('local server failed to start');
}

async function exportPdf(browser, mode, outFile) {
  const context = await browser.newContext({ viewport: { width: 1280, height: 1600 } });
  const page = await context.newPage();
  const url = `${baseUrl}${indexUrlPath}`;

  const errors = [];
  page.on('pageerror', e => errors.push(String(e)));
  page.on('console', m => { if (m.type() === 'error') errors.push('[console] ' + m.text()); });

  await page.goto(url, { waitUntil: 'networkidle' });
  await page.evaluate((m) => { document.body.dataset.pdf = m; }, mode);
  await page.emulateMedia({ media: 'print' });
  await page.waitForTimeout(800);

  await page.pdf({
    path: outFile,
    format: 'A3',
    landscape: true,
    printBackground: true,
    preferCSSPageSize: true,
    margin: { top: '12mm', right: '12mm', bottom: '12mm', left: '12mm' },
  });

  await context.close();
  return { errors };
}

(async () => {
  await fs.mkdir(distDir, { recursive: true });
  const server = await startServer();
  const browser = await chromium.launch();

  try {
    for (const f of flavors) {
      const outFile = path.join(distDir, f.file);
      log(`exporting ${f.label} → ${path.relative(projectRoot, outFile)}`);
      const { errors } = await exportPdf(browser, f.mode, outFile);
      if (errors.length) {
        log('⚠️  page errors:', errors);
      } else {
        log('✓ no page errors');
      }
      const stat = await fs.stat(outFile);
      log(`  size: ${(stat.size / 1024).toFixed(1)} KB`);
    }
  } finally {
    await browser.close();
    try { process.kill(-server.pid, 'SIGTERM'); } catch {}
    log('done');
  }
})();
