type Send = (message: object) => unknown;
type Tool = { name: string; description: string };
type Service = {id: string; label: string; base_url: string; paths: string[]; methods: string[]; auth: string; header: string};
type Plan = {goal?: string; steps?: {text: string; status: string}[]; next_step?: string};
type State = { type?: string; success?: boolean; message?: string; request_id?: string;
  tool_id?: string; tool_name?: string; content?: string; status?: string;
  preview_image?: string;
  memory_file?: string; persona_file?: string;
  settings?: Record<string, unknown>; tools?: Tool[]; events?: State[]; plan?: Plan };

export class RuntimePanel {
  private root = document.createElement("details");
  private log = document.createElement("div");
  private tools = document.createElement("div");
  private controls = new Map<string, HTMLInputElement>();
  private persona = "default";
  private services: Service[] = [];
  private serviceList = document.createElement("div");
  private planView = document.createElement("pre");
  private memoryFiles = document.createElement("p");
  constructor(private send: Send) {
    this.root.className = "runtime-panel";
    const summary = document.createElement("summary"); summary.textContent = "项目与工具";
    this.root.append(summary);
    const hint = document.createElement("p"); hint.className = "field-hint";
    hint.textContent = "项目目录位于当前角色的 workspace 内。留空使用整个角色工作区；填写子目录可缩小操作范围。";
    this.root.append(hint);
    this.field("project_folder", "项目子目录", "", "text");
    const toolsHint = document.createElement("p"); toolsHint.className = "field-hint";
    toolsHint.textContent = "所有已接入工具默认允许，由模型决定调用，不再逐次确认。项目文件仍限于所选目录；运行代码、浏览器及外部服务需要相应环境和连接信息。";
    this.root.append(toolsHint);
    this.field("temperature", "生成温度", "0.7", "number");
    this.field("max_tokens", "单次输出上限", "8192", "number");
    const memoryHint = document.createElement("p"); memoryHint.className = "field-hint";
    memoryHint.textContent = "人设和记忆使用 UTF-8 文本，保存后下一轮加载。新增记忆保留对话；局部删改会处理相关旧内容，清空整份记忆才重置上下文。原聊天档案仍可查看。记忆由当前聊天模型按积累量整理。";
    this.memoryFiles.className = "field-hint";
    this.root.append(memoryHint, this.memoryFiles);
    const save = document.createElement("button"); save.type = "button"; save.className = "secondary-button";
    save.textContent = "应用项目设置";
    save.onclick = () => this.send({type: "runtime-settings", settings: this.values()});
    const refresh = document.createElement("button"); refresh.type = "button"; refresh.className = "secondary-button";
    refresh.textContent = "刷新工具与记录"; refresh.onclick = () => this.send({type: "runtime-settings"});
    const stop = document.createElement("button"); stop.type = "button"; stop.className = "secondary-button";
    stop.textContent = "停止当前任务"; stop.onclick = () => this.send({type: "interrupt-signal", text: ""});
    this.root.append(save, refresh, stop, this.planView);
    this.serviceEditor();
    this.root.append(this.tools, this.log);
    document.querySelector("#settingsPanel")?.append(this.root);
    this.log.className = "runtime-log"; this.log.setAttribute("aria-live", "polite");
  }
  private field(key: string, label: string, value: string, type: string) {
    const row = document.createElement("label"); row.className = "field";
    const text = document.createElement("span"); text.textContent = label;
    const input = document.createElement("input"); input.type = type; input.value = value;
    if (key === "temperature") { input.min = "0"; input.max = "2"; input.step = "0.1"; }
    if (key === "max_tokens") { input.min = "512"; input.max = "65536"; }
    this.controls.set(key, input); row.append(text, input); this.root.append(row);
  }
  private values() {
    return {...Object.fromEntries([...this.controls].map(([key, input]) => [key, input.value])), services: this.services};
  }
  private savedSettings(value: Record<string, unknown> = {}) {
    // Only active fields survive older browser snapshots with ask/forbid flags.
    return {project_folder: value.project_folder ?? "", temperature: value.temperature ?? 0.7,
      max_tokens: value.max_tokens ?? 8192, services: value.services ?? []};
  }
  private serviceEditor() {
    const section = document.createElement("details");
    const title = document.createElement("summary"); title.textContent = "连接服务与 API"; section.append(title);
    const hint = document.createElement("p"); hint.className = "field-hint";
    hint.textContent = "可连接电脑上的程序、局域网设备或公共 API。填写你确认的地址与允许路径；模型自行决定何时调用。密钥在下面单独保存，不填进聊天。";
    section.append(hint, this.serviceList);
    const inputs: Record<string, HTMLInputElement> = {};
    for (const [key, label, value] of [["id", "服务 ID（英文）", ""], ["label", "名称", ""],
      ["base_url", "地址（协议、主机、端口）", ""], ["paths", "允许路径（逗号分隔）", "/api/"],
      ["methods", "允许方法（逗号分隔）", "GET,HEAD"], ["header", "自定义认证请求头", "Authorization"]]) {
      const row = document.createElement("label"); row.className = "field";
      const span = document.createElement("span"); span.textContent = label;
      const input = document.createElement("input"); input.value = value; inputs[key] = input;
      row.append(span, input); section.append(row);
    }
    const auth = document.createElement("select");
    for (const [value, label] of [["none", "无需密钥"], ["bearer", "Bearer Token"], ["header", "自定义请求头密钥"]]) {
      const option = document.createElement("option"); option.value = value; option.textContent = label; auth.append(option);
    }
    const authLabel = document.createElement("label"); authLabel.className = "field";
    const authText = document.createElement("span"); authText.textContent = "认证方式"; authLabel.append(authText, auth); section.append(authLabel);
    const save = document.createElement("button"); save.type = "button"; save.textContent = "添加或更新服务";
    save.onclick = () => {
      const service: Service = {id: inputs.id.value.trim(), label: inputs.label.value.trim(), base_url: inputs.base_url.value.trim(),
        paths: inputs.paths.value.split(",").map(s => s.trim()).filter(Boolean), methods: inputs.methods.value.toUpperCase().split(",").map(s => s.trim()).filter(Boolean), auth: auth.value, header: inputs.header.value.trim()};
      this.send({type: "runtime-settings", settings: {...this.values(), services: [...this.services.filter(s => s.id !== service.id), service]}});
    };
    const secret = document.createElement("input"); secret.type = "password"; secret.autocomplete = "new-password";
    secret.placeholder = "先应用服务，再在此输入密钥"; secret.setAttribute("aria-label", "服务密钥");
    const saveSecret = document.createElement("button"); saveSecret.type = "button"; saveSecret.textContent = "安全保存密钥";
    saveSecret.onclick = () => { this.send({type: "pc-service-token", service_id: inputs.id.value.trim(), secret: secret.value}); secret.value = ""; };
    const clearSecret = document.createElement("button"); clearSecret.type = "button"; clearSecret.textContent = "清除密钥";
    clearSecret.onclick = () => this.send({type: "pc-service-token", service_id: inputs.id.value.trim(), clear: true});
    section.append(save, secret, saveSecret, clearSecret);
    this.root.append(section);
    this.serviceList.addEventListener("click", event => {
      const target = event.target as HTMLElement;
      const service = this.services.find(s => s.id === target.dataset.edit);
      if (!service) return;
      for (const [key, input] of Object.entries(inputs)) input.value = key === "paths" ? service.paths.join(",") : key === "methods" ? service.methods.join(",") : String(service[key as keyof Service]);
      auth.value = service.auth;
    });
  }
  private renderServices() {
    this.serviceList.replaceChildren();
    for (const service of this.services) {
      const row = document.createElement("div");
      const edit = document.createElement("button"); edit.type = "button"; edit.dataset.edit = service.id;
      edit.textContent = `${service.label} · ${service.base_url}`;
      const remove = document.createElement("button"); remove.type = "button"; remove.textContent = "移除连接";
      remove.onclick = () => this.send({type: "runtime-settings", settings: {...this.values(), services: this.services.filter(s => s.id !== service.id)}});
      row.append(edit, remove); this.serviceList.append(row);
    }
  }
  private renderPlan(plan?: Plan) {
    const status: Record<string, string> = {pending: "待处理", in_progress: "进行中", completed: "已完成", blocked: "待补充条件"};
    this.planView.textContent = plan?.goal ? `当前任务：${plan.goal}\n${(plan.steps || []).map(step => `${status[step.status] || step.status}：${step.text}`).join("\n")}\n下一步：${plan.next_step || ""}` : "";
    this.planView.className = "runtime-plan";
  }
  connect(persona: string) {
    this.persona = persona;
    this.memoryFiles.textContent = "";
    this.log.replaceChildren();
    this.renderPlan();
    this.root.querySelectorAll<HTMLInputElement>('input[type="password"]').forEach(input => input.value = "");
    try {
      const value = localStorage.getItem(`melomate-runtime:${persona}`);
      this.send({type: "runtime-settings", settings: this.savedSettings(value ? JSON.parse(value) : {})});
    } catch { this.send({type: "runtime-settings", settings: this.savedSettings()}); }
  }
  handle(value: unknown): boolean {
    const message = value as State;
    if (message.type === "pc-service-token-result") { this.entry(message.message || "凭据状态已更新"); return true; }
    if (message.type === "work-plan") { this.renderPlan(message.plan); return true; }
    if (message.type === "runtime-state") {
      if (!message.success) { this.entry(message.message || "设置失败"); return true; }
      this.memoryFiles.textContent = [message.persona_file ? `人设：characters/profiles/${message.persona_file}` : "", message.memory_file ? `记忆：${message.memory_file}` : ""].filter(Boolean).join("；");
      if (message.settings) {
        for (const [key, input] of this.controls) {
          if (input instanceof HTMLInputElement && input.type === "checkbox") input.checked = message.settings[key] === true;
          else input.value = String(message.settings[key] ?? "");
        }
        this.services = (message.settings.services || []) as Service[];
        this.renderServices();
        try { localStorage.setItem(`melomate-runtime:${this.persona}`, JSON.stringify(this.savedSettings(message.settings))); } catch { /* optional persistence */ }
      }
      this.tools.replaceChildren();
      this.renderPlan(message.plan);
      for (const tool of message.tools || []) {
        const row = document.createElement("div"); row.className = "field";
        const name = document.createElement("span"); name.textContent = tool.name; name.title = tool.description;
        row.append(name); this.tools.append(row);
      }
      this.log.replaceChildren();
      for (const event of message.events || []) this.entry(`${event.tool_name} · ${event.status}\n${event.content || ""}`);
      this.entry("项目与工具设置已同步。"); return true;
    }
    if (message.type === "tool-approval-request" || message.type === "tool-approval-closed") {
      if (message.type === "tool-approval-request") this.entry("前后端版本不一致，请重启 MeloMate 并刷新页面以使用默认允许模式。");
      return true;
    }
    if (message.type === "tool_call_status") {
      this.entry(`${message.tool_name} · ${message.status}\n${message.content || ""}`);
      if (message.preview_image && /^data:image\/(jpeg|png);base64,[A-Za-z0-9+/=]+$/.test(message.preview_image) && message.preview_image.length < 12000000) {
        const preview = document.createElement("img"); preview.src = message.preview_image; preview.alt = "工具返回的浏览器截图";
        preview.style.maxWidth = "100%"; this.log.append(preview);
      }
      return true;
    }
    return false;
  }
  private entry(text: string) {
    const p = document.createElement("pre"); p.textContent = text.slice(0, 2000); this.log.append(p);
    while (this.log.childElementCount > 40) this.log.firstElementChild?.remove();
  }
}
