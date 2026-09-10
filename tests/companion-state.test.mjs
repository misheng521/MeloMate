import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import ts from 'typescript';

// Execute the actual frontend functions with browser/transport substitutes.
const source = readFileSync(new URL('../src/main.ts', import.meta.url), 'utf8');
const tree = ts.createSourceFile('main.ts', source, ts.ScriptTarget.Latest, true);
const names = new Set(['canTriggerProactiveSpeak', 'requestProactiveSpeak',
  'completeProactiveTurn', 'pollCompanionState', 'handleWsMessage',
  'resetDisconnectedConversation', 'stopCurrentResponsePlayback', 'cancelUserInputPriority']);
const functions = tree.statements.filter(node => ts.isFunctionDeclaration(node) && names.has(node.name?.text));
assert.equal(functions.length, names.size);
const js = ts.transpileModule(functions.map(node => node.getText(tree)).join('\n'), {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None },
}).outputText;

function fixture() {
  const sent = [];
  const audio = () => ({ paused: true, pause() {}, removeAttribute() {}, load() {} });
  const state = { isWsReady: true, isCapturing: false, isAssistantResponding: false, thinking: false,
    responseAudio: audio(), voiceChatAudio: audio(), avatarDriver: { stopLipSync() {} },
    audioQueueVersion: 0,
    isUserSpeaking: false, isUserInputPriorityActive: false,
    currentProactiveTurnId: '', currentProactiveIsAutomatic: false,
    currentCompanionUid: 'Alice', proactiveSpeakToggle: { checked: true },
    proactiveUnansweredCount: 0, nextProactiveSpeakAt: 0,
    previousProactiveSpeakAt: 123, lastProactiveSpeakAt: 456,
    lastUserConversationActivityAt: Date.now(), latestScreenImage: null,
    runtimePanel: { handle: () => false }, screenVisionEnabled: () => false,
    screenVisionConfigPayload: () => ({}), screenImagesForNextTurn: async () => [],
    proactiveTurnId: () => 'event-turn', proactiveBaseIntervalMs: () => 120000,
    scheduleNextProactiveSpeak: () => {}, resetProactiveSilenceEpisode: () => {},
    appendLine: () => {}, syncProactiveSpeakButton: () => {}, console,
    sendWs: message => { sent.push(message); return true; },
  };
  state.setThinking = value => { state.thinking = value; };
  vm.createContext(state);
  vm.runInContext(js, state);
  return { state, sent };
}
const event = { type: 'companion-state', success: true, conf_uid: 'Alice',
  event_ready: true, event_token: 'server-token' };
const flush = () => new Promise(resolve => setImmediate(resolve));

test('text-only connection can react through the normal chat request', async () => {
  const { state, sent } = fixture();
  state.handleWsMessage(event);
  await flush();
  assert.equal(sent.length, 1);
  assert.equal(sent[0].type, 'ai-speak-signal');
  assert.equal(sent[0].event_token, 'server-token');
  assert.equal(state.proactiveUnansweredCount, 1);
});

test('disabled switch, busy user, disconnected client and another role do not react', async () => {
  for (const override of [{ proactiveSpeakToggle: { checked: false } },
    { isAssistantResponding: true }, { isUserSpeaking: true },
    { isUserInputPriorityActive: true }, { isWsReady: false },
    { currentCompanionUid: 'Bob' }]) {
    const { state, sent } = fixture();
    Object.assign(state, override);
    state.handleWsMessage(event);
    await flush();
    assert.equal(sent.length, 0);
  }
});

test('switching off during image capture cancels an unsent opportunity', async () => {
  const { state, sent } = fixture();
  let release;
  state.screenImagesForNextTurn = () => new Promise(resolve => { release = resolve; });
  const pending = state.requestProactiveSpeak('automatic', false, 'token');
  state.proactiveSpeakToggle.checked = false;
  release([]);
  await pending;
  assert.equal(sent.length, 0);
  assert.equal(state.currentProactiveTurnId, '');
  assert.equal(state.thinking, false);
});

test('an obsolete opportunity cannot clear the thinking state of a new user turn', async () => {
  const { state, sent } = fixture();
  let release;
  state.screenImagesForNextTurn = () => new Promise(resolve => { release = resolve; });
  const pending = state.requestProactiveSpeak('automatic', false, 'token');
  state.currentProactiveTurnId = '';
  state.isUserInputPriorityActive = true;
  state.setThinking(true);
  release([]);
  await pending;
  assert.equal(sent.length, 0);
  assert.equal(state.thinking, true);
});

test('a skipped event restores counters and releases only its own turn', async () => {
  const { state } = fixture();
  await state.requestProactiveSpeak('automatic', false, 'token');
  state.handleWsMessage({ type: 'event-opportunity-skipped', turn_id: 'other' });
  assert.equal(state.thinking, true);
  state.handleWsMessage({ type: 'event-opportunity-skipped', turn_id: 'event-turn' });
  assert.equal(state.proactiveUnansweredCount, 0);
  assert.equal(state.lastProactiveSpeakAt, 456);
  assert.equal(state.currentProactiveTurnId, '');
  assert.equal(state.thinking, false);
});

test('disconnect releases text-only response and user-priority state for reconnection', async () => {
  const { state } = fixture();
  await state.requestProactiveSpeak('automatic', false, 'token');
  state.isAssistantResponding = true;
  state.isUserInputPriorityActive = true;
  state.activeAssistantTurnId = 'old-turn';
  state.isWsReady = false;
  state.resetDisconnectedConversation();
  assert.equal(state.thinking, false);
  assert.equal(state.isAssistantResponding, false);
  assert.equal(state.isUserInputPriorityActive, false);
  assert.equal(state.currentProactiveTurnId, '');
  assert.equal(state.activeAssistantTurnId, '');
  state.isWsReady = true;
  assert.equal(state.canTriggerProactiveSpeak(), true);
});

test('state polling sends channel flags without creating a conversation', () => {
  const { state, sent } = fixture();
  state.pollCompanionState();
  assert.deepEqual(JSON.parse(JSON.stringify(sent)), [{ type: 'companion-state-request', state: {
    proactive_enabled: true, microphone_active: false, screen_shared: false,
  } }]);
  state.isWsReady = false;
  state.pollCompanionState();
  assert.equal(sent.length, 1);
});
