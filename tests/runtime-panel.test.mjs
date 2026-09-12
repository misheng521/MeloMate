import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';

const code = ts.transpileModule(readFileSync(new URL('../src/runtime-panel.ts', import.meta.url), 'utf8'), {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
}).outputText;

class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.dataset = {}; this.style = {}; this.value = ''; this.textContent = ''; this.listeners = {}; this.scrollTop = 0; this.classList = {add() {}, remove() {}}; }
  append(...children) { this.children.push(...children); children.forEach(child => { child.parent = this; }); }
  replaceChildren(...children) { this.children = []; this.append(...children); }
  setAttribute() {}
  addEventListener(name, callback) { this.listeners[name] = callback; }
  focus() { this.focused = true; }
  get childElementCount() { return this.children.length; }
  get firstElementChild() { return this.children[0]; }
  remove() { this.parent.children = this.parent.children.filter(child => child !== this); }
  querySelectorAll(selector) {
    return this.children.flatMap(child => [child, ...child.querySelectorAll('*')])
      .filter(child => selector === '*' || child.tag === selector || (selector.startsWith('.') && child.className === selector.slice(1)) || (selector === 'input[type="password"]' && child.type === 'password'));
  }
}
class Input extends Element { constructor() { super('input'); } }
function fixture(saved) {
  const root = new Element('root'), settingsRoot = new Element('settings'), sent = [], storage = new Map();
  if (saved) storage.set('melomate-runtime:Alice', JSON.stringify(saved));
  const context = vm.createContext({ exports: {}, HTMLInputElement: Input,
    document: { createElement: tag => tag === 'input' ? new Input() : new Element(tag),
      querySelector: selector => selector === '.text-panel' ? root : settingsRoot },
    requestAnimationFrame: callback => callback(),
    localStorage: { getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value) },
  });
  vm.runInContext(code, context);
  const panel = new context.exports.RuntimePanel(message => sent.push(JSON.parse(JSON.stringify(message))));
  return { panel, root, settingsRoot, sent, storage };
}

test('removed settings stay removed while existing project scope and connections survive reconnect', () => {
  const settings = { project_folder: 'projects/demo', temperature: 0.8, max_tokens: 4096,
    services: [{ id: 'local', label: 'Local', base_url: 'http://127.0.0.1:9000', paths: ['/api/'], methods: ['GET'], auth: 'none', header: '' }],
    workspace: 'forbid', browser: 'ask', tools: { read_memory: 'forbid' } };
  const { panel, root, settingsRoot, sent, storage } = fixture(settings);
  panel.connect('Alice');
  const expected = { project_folder: settings.project_folder, services: settings.services };
  assert.deepEqual(sent[0], { type: 'runtime-settings', settings: expected });
  panel.handle({ type: 'runtime-state', success: true, settings,
    tools: [{ name: 'read_memory', description: 'Read notes', permission: 'forbid' }] });
  assert.deepEqual(JSON.parse(storage.get('melomate-runtime:Alice')), expected);
  assert.equal(settingsRoot.childElementCount, 0);
  assert.equal(root.querySelectorAll('input').length, 0);
  assert.equal(root.querySelectorAll('select').length, 0);
  assert.equal(root.firstElementChild.hidden, true);
  panel.begin('turn1');
  panel.handle({type: 'tool_call_status', turn_id: 'turn1', tool_name: 'write_file', status: 'running'});
  const line = new Element('p'); panel.bindReply(line, 'turn1', '完成了', 'Alice：');
  assert.equal(root.firstElementChild.hidden, true);
  line.querySelectorAll('.reply-detail-trigger')[0].ondblclick();
  assert.equal(root.firstElementChild.hidden, false);
  root.querySelectorAll('button').find(button => button.textContent === '停止当前任务').onclick();
  assert.deepEqual(sent.at(-1), { type: 'interrupt-signal', text: '' });
  panel.connect('Bob');
  assert.equal(root.firstElementChild.hidden, true);
  assert.deepEqual(sent.at(-1).settings, {project_folder: '', services: []});
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
    assert.deepEqual(sent[0].settings, { project_folder: '', services: [] });
  }
});

function allText(node) { return node.textContent + node.children.map(allText).join(''); }

test('double click opens only that reply, renders reasoning as text, and back preserves chat', () => {
  const {panel, root} = fixture();
  panel.begin('one');
  panel.handle({type: 'reasoning_delta', turn_id: 'one', text: '<script>reason one</script>'});
  panel.handle({type: 'tool_call_status', turn_id: 'one', tool_name: 'read_file', status: 'completed', content: 'first result'});
  const line = new Element('p'); panel.bindReply(line, 'one', 'answer one', 'Alice：');
  panel.begin('two');
  panel.handle({type: 'reasoning_delta', turn_id: 'two', text: 'reason two'});
  panel.handle({type: 'reasoning_delta', turn_id: 'one', text: 'late old event'});
  line.querySelectorAll('.reply-detail-trigger')[0].ondblclick();
  assert.match(allText(root), /reason one/);
  assert.match(allText(root), /first result/);
  assert.doesNotMatch(allText(root), /reason two|late old event/);
  assert.equal(root.querySelectorAll('script').length, 0);
  assert.equal(root.querySelectorAll('button').find(b => b.textContent === '停止当前任务').hidden, true);
  root.querySelectorAll('button').find(b => b.textContent === '← 返回').onclick();
  assert.equal(root.firstElementChild.hidden, true);
  assert.match(allText(line), /answer one/);
  assert.equal(line.querySelectorAll('.reply-detail-trigger')[0].focused, true);
});

test('open details update live, absent reasoning is explicit, unscoped logs never attach', () => {
  const {panel, root} = fixture();
  panel.begin('one');
  const line = new Element('p'); panel.bindReply(line, 'one', 'answer', 'Alice：');
  line.querySelectorAll('.reply-detail-trigger')[0].listeners.keydown({key: 'Enter', preventDefault() {}});
  panel.handle({type: 'tool_call_status', tool_name: 'unscoped', content: 'wrong'});
  panel.handle({type: 'tool_call_status', turn_id: 'one', tool_name: 'browser_read', content: 'real', preview_image: 'javascript:alert(1)'});
  assert.match(allText(root), /real/);
  assert.doesNotMatch(allText(root), /wrong/);
  assert.equal(root.querySelectorAll('img').length, 0);
  panel.finish('one');
  assert.match(allText(root), /未返回可展示的思考内容/);
  panel.connect('Bob');
  line.querySelectorAll('.reply-detail-trigger')[0].ondblclick();
  assert.equal(root.firstElementChild.hidden, true);
});

test('bounded details retain recent replies instead of evicting the newest one', () => {
  const {panel, root} = fixture();
  const lines = [];
  for (let i = 0; i < 122; i++) {
    panel.begin(String(i));
    panel.handle({type: 'reasoning_delta', turn_id: String(i), text: `trace-${i}`});
    const line = new Element('p'); panel.bindReply(line, String(i), 'answer', 'Alice：'); lines.push(line);
  }
  lines[120].querySelectorAll('.reply-detail-trigger')[0].ondblclick();
  assert.match(allText(root), /trace-120/);
  lines[0].querySelectorAll('.reply-detail-trigger')[0].ondblclick();
  assert.match(allText(root), /没有可用的详情记录/);
});
