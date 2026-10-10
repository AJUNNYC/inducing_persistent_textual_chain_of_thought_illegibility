import { JSDOM } from 'jsdom';
import { readdir, readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
const dom = new JSDOM('<!doctype html><html><body></body></html>');
globalThis.window = dom.window;
globalThis.document = dom.window.document;
const { default: mermaid } = await import('mermaid');
mermaid.initialize({ startOnLoad: false, securityLevel: 'strict' });
const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
async function* walk(path) {
  for (const e of await readdir(path, { withFileTypes: true })) {
    const p = join(path, e.name);
    if (e.isDirectory()) yield* walk(p);
    else if (e.name.endsWith('.mmd')) yield p;
  }
}
let passed = 0, failed = 0;
for await (const path of walk(join(root, 'spec/mermaid'))) {
  try { await mermaid.parse(await readFile(path, 'utf8')); passed++; }
  catch (error) { failed++; process.stderr.write(`${path}: ${String(error.message).slice(0,500)}\n`); }
}
process.stdout.write(JSON.stringify({ parser: 'mermaid@11.12.1', passed, failed })+'\n');
if (failed) process.exitCode = 1;
