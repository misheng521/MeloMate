type Send = (message: object) => unknown;
type Plan = {goal?: string; steps?: {text: string; status: string}[]; next_step?: string};
type Event = {tool_name?: string; status?: string; content?: string; preview_image?: string};
type State = Event & {type?: string; turn_id?: string; text?: string; success?: boolean;
  message?: string; settings?: Record<string, unknown>; plan?: Plan};
type Trace = {id: string; reasoning: string; truncated: boolean; events: Event[]; plan?: Plan; finished: boolean};

export class RuntimePanel {
  private root = document.createElement("section");
  private body = document.createElement("div");
  private stop = document.createElement("button");
  private heading = document.createElement("h2");
  private traces = new Map<string, Trace>();
  private persona = "default";
  private activeId = "";
  private selectedId = "";
  private returnFocus: HTMLElement | null = null;
  private redrawPending = false;
  constructor(private send: Send) {
    this.root.className = "reply-details";
    this.root.hidden = true;
    this.root.setAttribute("aria-label", "回复详情");
    const header = document.createElement("div"); header.className = "reply-details-header";
    const back = document.createElement("button"); back.type = "button"; back.className = "secondary-button";
    back.textContent = "← 返回"; back.onclick = () => this.close();
    this.heading.textContent = "回复详情"; this.heading.tabIndex = -1;
    this.stop.type = "button"; this.stop.className = "secondary-button"; this.stop.textContent = "停止当前任务";
    this.stop.onclick = () => this.send({type: "interrupt-signal", text: ""});
    header.append(back, this.heading, this.stop);
    this.body.className = "reply-details-body";
    this.root.append(header, this.body);
    this.root.addEventListener("keydown", event => { if (event.key === "Escape") this.close(); });
    document.querySelector(".text-panel")?.append(this.root);
  }
  private savedSettings(value: Record<string, unknown> = {}) {
    return {project_folder: value.project_folder ?? "", services: value.services ?? []};
  }
  connect(persona: string) {
    if (persona !== this.persona) { this.close(); this.traces.clear(); }
    this.finish(this.activeId);
    this.activeId = "";
    this.persona = persona;
    try {
      const value = localStorage.getItem(`melomate-runtime:${persona}`);
      this.send({type: "runtime-settings", settings: this.savedSettings(value ? JSON.parse(value) : {})});
    } catch { this.send({type: "runtime-settings", settings: this.savedSettings()}); }
  }
  begin(id: string) {
    if (!id) return;
    this.finish(this.activeId);
    this.activeId = id;
    if (!this.traces.has(id)) this.traces.set(id, {id, reasoning: "", truncated: false, events: [], finished: false});
    this.prune();
  }
  finish(id: string) {
    const trace = this.traces.get(id);
    if (trace) { trace.finished = true; this.scheduleRender(); }
  }
  disconnect() { this.finish(this.activeId); this.activeId = ""; }
  bindReply(line: HTMLElement, id: string, text: string, prefix: string) {
    // An inline semantic button preserves the original line wrapping.
    const button = document.createElement("span");
    button.setAttribute("role", "button"); button.tabIndex = 0;
    button.className = "reply-detail-trigger"; button.textContent = text;
    button.title = "双击查看这条回复的思考与工具记录";
    button.setAttribute("aria-label", "查看回复详情：" + text.slice(0, 80));
    const persona = this.persona;
    const open = () => { if (persona === this.persona) this.open(id, button); };
    button.ondblclick = open;
    button.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); open(); }
    });
    const label = document.createElement("span"); label.textContent = prefix;
    line.replaceChildren(label, button);
  }
  private open(id: string, source: HTMLElement) {
    this.selectedId = id;
    this.returnFocus = source;
    this.root.hidden = false;
    document.querySelector(".text-panel")?.classList.add("show-reply-details");
    this.render(); this.heading.focus();
  }
  private close() {
    this.root.hidden = true;
    this.selectedId = "";
    document.querySelector(".text-panel")?.classList.remove("show-reply-details");
    this.returnFocus?.focus(); this.returnFocus = null;
  }
  handle(value: unknown): boolean {
    const message = value as State;
    if (message.type === "runtime-state") {
      if (message.success && message.settings) {
        try { localStorage.setItem(`melomate-runtime:${this.persona}`, JSON.stringify(this.savedSettings(message.settings))); } catch { /* optional persistence */ }
      }
      // Global restored logs do not belong to a specific reply.
      return true;
    }
    if (message.type === "pc-service-token-result" || message.type === "tool-approval-request" || message.type === "tool-approval-closed") return true;
    if (!["reasoning_delta", "tool_call_status", "work-plan"].includes(message.type || "")) return false;
    // Ignore unscoped and late events instead of assigning them to another reply.
    if (!message.turn_id || message.turn_id !== this.activeId) return true;
    const trace = this.traces.get(message.turn_id);
    if (!trace || trace.finished) return true;
    if (message.type === "reasoning_delta") {
      const text = typeof message.text === "string" ? message.text : "";
      const room = Math.max(0, 250000 - trace.reasoning.length);
      trace.reasoning += text.slice(0, room);
      trace.truncated ||= text.length > room;
    } else if (message.type === "work-plan") {
      trace.plan = message.plan;
    } else {
      const event: Event = {tool_name: String(message.tool_name || "工具"), status: String(message.status || ""), content: String(message.content || "").slice(0, 4000)};
      if (message.preview_image && message.preview_image.length < 4000000 && /^data:image\/(jpeg|png);base64,[A-Za-z0-9+/=]+$/.test(message.preview_image)) {
        for (const previous of trace.events) delete previous.preview_image;
        event.preview_image = message.preview_image;
      }
      trace.events.push(event);
      trace.events = trace.events.slice(-100);
    }
    this.prune(); this.scheduleRender(); return true;
  }
  private prune() {
    let remaining = 8000000, kept = 0;
    for (const trace of [...this.traces.values()].reverse()) {
      const size = trace.reasoning.length + JSON.stringify(trace.events).length;
      if ((size > remaining || kept >= 120) && trace.id !== this.activeId && trace.id !== this.selectedId) {
        this.traces.delete(trace.id);
      } else { remaining -= size; kept++; }
    }
  }
  private scheduleRender() {
    if (this.root.hidden || this.redrawPending) return;
    this.redrawPending = true;
    requestAnimationFrame(() => { this.redrawPending = false; if (!this.root.hidden) this.render(); });
  }
  private section(title: string, text: string) {
    const heading = document.createElement("h3"); heading.textContent = title;
    const content = document.createElement("pre"); content.textContent = text;
    this.body.append(heading, content);
  }
  private render() {
    const scroll = this.body.scrollTop;
    this.body.replaceChildren();
    const trace = this.traces.get(this.selectedId);
    this.stop.hidden = !trace || trace.finished || trace.id !== this.activeId;
    if (!trace) {
      this.section("暂无记录", "这条回复没有可用的详情记录，可能来自旧会话或记录已清理。");
      return;
    }
    this.section("思考内容", trace.reasoning || (trace.finished ? "本次 API 未返回可展示的思考内容。" : "正在等待 API 返回思考内容……"));
    if (trace.truncated) this.section("显示提示", "这条思考内容较长，详情页只显示前 250,000 字符；此显示限制不截断 API 的协议回传。");
    if (trace.plan?.goal) {
      const labels: Record<string, string> = {pending: "待处理", in_progress: "进行中", completed: "已完成", blocked: "待补充条件"};
      this.section("任务计划", `${trace.plan.goal}\n${(trace.plan.steps || []).map(step => `${labels[step.status] || step.status}：${step.text}`).join("\n")}\n${trace.plan.next_step || ""}`);
    }
    this.section("工具记录", trace.events.length ? "本条回复最近的工具调用与结果：" : "这条回复暂时没有工具调用记录。");
    const labels: Record<string, string> = {running: "执行中", completed: "已完成", error: "失败"};
    for (const event of trace.events) {
      this.section(`${event.tool_name} · ${labels[event.status || ""] || event.status}`, event.content || "");
      if (event.preview_image) {
        const img = document.createElement("img"); img.src = event.preview_image; img.alt = "工具返回的截图";
        this.body.append(img);
      }
    }
    this.body.scrollTop = scroll;
  }
}
