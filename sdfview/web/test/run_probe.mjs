// run_probe.mjs — runs probe jobs in headless Chromium (Playwright) against a running server; used by
// tests/sdfview/test_hosts.py. stdin: {"base": url, "threeDir": dir or null, "jobs": [{host, scene, field, queries,
// settings, kind: "probe"|"render"}]}; stdout: {"results": [...]} (per job: values, or {"error": message}).
// WebGPU: Chromium exposes it on localhost with --enable-unsafe-webgpu (SwiftShader without a GPU).
import { createRequire } from 'module';
import { execSync } from 'child_process';
import fs from 'fs';
import path from 'path';

function loadPlaywright() {
  const roots = [process.cwd() + '/', execSync('npm root -g').toString().trim() + '/'];
  for (const r of roots) { try { return createRequire(r)('playwright'); } catch (e) { /* next */ } }
  throw new Error('playwright not found (npm install -g playwright)');
}

const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const { chromium } = loadPlaywright();
// lazy_clear_resource_on_first_use off: Chromium 141's Dawn zero-fills a partially written 3D texture with
// RENDER_ATTACHMENT usage through a 2D view, a validation error; three.js uploads 3D textures one slice at a time
// with that usage. Every host writes its textures completely, so the zero fill has nothing to change here.
const browser = await chromium.launch({
  headless: true,
  args: ['--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--disable-dawn-features=lazy_clear_resource_on_first_use'],
});
const results = [];
const pages = {};
for (const job of input.jobs) {
  try {
    if (!pages[job.host]) {
      const page = await browser.newPage();
      const logs = [];
      page.on('console', (m) => logs.push(m.text()));
      page.on('pageerror', (e) => logs.push(String(e)));
      if (input.threeDir) {  // serve three.js from a local copy instead of the CDN
        await page.route('https://cdn.jsdelivr.net/npm/three@*/**', (route) => {
          const rel = new URL(route.request().url()).pathname.replace(/^\/npm\/three@[^/]+\//, '');
          route.fulfill({ path: path.join(input.threeDir, rel), contentType: 'text/javascript' });
        });
      }
      await page.goto(`${input.base}/web/test/probe.html?host=${job.host}`);
      const ready = await page.waitForFunction(() => window.sdfxReady === true || window.sdfxError, null, { timeout: 60000 })
        .then(() => page.evaluate(() => window.sdfxError || null))
        .catch(() => 'timeout');
      if (ready) throw new Error(`page not ready: ${ready} | ${logs.join(' | ')}`);
      pages[job.host] = { page, logs };
    }
    const { page, logs } = pages[job.host];
    const t0 = Date.now();
    const value = job.kind === 'render'
      ? await page.evaluate(([s, o]) => window.sdfxRender(s, o), [job.scene, job.settings || {}])
      : await page.evaluate(([s, f, q, o]) => window.sdfxProbe(s, f, q, o), [job.scene, job.field, job.queries, job.settings || {}]);
    results.push({ value, seconds: (Date.now() - t0) / 1000, logs: logs.slice(-5) });
  } catch (e) {
    results.push({ error: String(e).slice(0, 2000), logs: pages[job.host]?.logs.slice(-10) });
  }
}
await browser.close();
process.stdout.write(JSON.stringify({ results }));
