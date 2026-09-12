import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';

const code = ts.transpileModule(readFileSync(new URL('../src/runtime-panel.ts', import.meta.url), 'utf8'), {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
}).outputText;

class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.dataset = {}; this.style = {}; this.value = ''; this.textContent = ''; }
  append(...children) { this.children.push(...children); children.forEach(child => { child.parent = this; }); }
  replaceChildren(...children) { this.children = []; this.append(...children); }
  setAttribute() {}
  addEventListener() {}
  get childElementCount() { return this.children.length; }
  get firstElementChild() { return this.children[0]; }
  remove() { this.parent.children = this.parent.children.filter(child => child !== this); }
  querySelectorAll(selector) {
    return this.children.flatMap(child => [child, ...child.querySelectorAll('*')])
      .filter(child => selector === '*' || child.tag === selector || (selector === 'input[type="password"]' && child.type === 'password'));
  }
}
class Input extends Element { constructor() { super('input'); } }
function fixture(saved) {
  const root = new Element('root'), sent = [], storage = new Map();
  if (saved) storage.set('melomate-runtime:Alice', JSON.stringify(saved));
  const context = vm.createContext({ exports: {}, HTMLInputElement: Input,
    document: { createElement: tag => tag === 'input' ? new Input() : new Element(tag), querySelector: () => root },
    localStorage: { getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value) },
  });
  vm.runInContext(code, context);
  const panel = new context.exports.RuntimePanel(message => sent.push(JSON.parse(JSON.stringify(message))));
  return { panel, root, sent, storage };
}

test('old permissions disappear while project, model settings and services survive reconnect', () => {
  const settings = { project_folder: 'projects/demo', temperature: 0.8, max_tokens: 4096,
    services: [{ id: 'local', label: 'Local', base_url: 'http://127.0.0.1:9000', paths: ['/api/'], methods: ['GET'], auth: 'none', header: '' }],
    workspace: 'forbid', browser: 'ask', tools: { read_memory: 'forbid' } };
  const { panel, root, sent, storage } = fixture(settings);
  panel.connect('Alice');
  const expected = { project_folder: settings.project_folder, temperature: 0.8, max_tokens: 4096, services: settings.services };
  assert.deepEqual(sent[0], { type: 'runtime-settings', settings: expected });
  panel.handle({ type: 'runtime-state', success: true, settings,
    tools: [{ name: 'read_memory', description: 'Read notes', permission: 'forbid' }] });
  assert.deepEqual(JSON.parse(storage.get('melomate-runtime:Alice')), expected);
  // The remaining select is service authentication, never a tool permission.
  const selects = root.querySelectorAll('select');
  assert.equal(selects.length, 1);
  assert.deepEqual(selects[0].children.map(option => option.value), ['none', 'bearer', 'header']);
  root.querySelectorAll('button').find(button => button.textContent === '应用项目设置').onclick();
  assert.equal(sent.at(-1).settings.project_folder, 'projects/demo');
  assert.equal('tools' in sent.at(-1).settings, false);
  root.querySelectorAll('button').find(button => button.textContent === '停止当前任务').onclick();
  assert.deepEqual(sent.at(-1), { type: 'interrupt-signal', text: '' });
});

test('obsolete approval messages never get silently approved or create choice controls', () => {
  const { panel, root, sent } = fixture();
  const before = root.querySelectorAll('button').length;
  assert.equal(panel.handle({ type: 'tool-approval-request', request_id: 'legacy', tool_name: 'write' }), true);
  assert.equal(root.querySelectorAll('button').length, before);
  assert.equal(sent.length, 0);
});

test('new or malformed browser settings use ordinary application defaults', () => {
  for (const saved of [undefined, null]) {
    const { panel, sent, storage } = fixture(saved);
    if (saved === null) storage.set('melomate-runtime:Alice', '{bad');
    panel.connect('Alice');
    assert.deepEqual(sent[0].settings, { project_folder: '', temperature: 0.7, max_tokens: 8192, services: [] });
  }
});
