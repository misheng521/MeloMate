# 角色记忆目录

MeloMate 会在这里为每个角色创建独立目录，例如：

```text
characters/memory/avatar_example_001/
  core_memory.json
  short_memory.json
  core_memory.json.bak
  short_memory.json.bak
  backups/
```

这些文件包含真实对话和用户记忆，已被 `.gitignore` 排除，不会提交到 GitHub。仓库只保存本说明和 `core_memory.example.json` 空白示例。

## 手工编辑

建议先停止 MeloMate，再用 UTF-8 编辑 `core_memory.json`。程序也会在下一次读取时热加载合法 JSON；如果保存到一半或 JSON 语法错误，程序会先把错误内容保留为 `core_memory.invalid-时间.json`，然后从 `.bak` 或空白结构恢复，避免聊天服务无法启动。

- 数组里直接写字符串，例如 `"桂花糕"`，会被当作 `source: "manual"`。
- 手工记录的优先级最高，模型不能删除、降级或覆盖。
- 若编辑完整记录对象，请把 `source` 写成 `"manual"`，并保持 `status: "active"`。
- `pending_inferences` 是模型尚未得到两次独立用户证据的候选记忆。
- `forgotten_topics` 是用户要求忘记的主题；模型不能从旧对话重新添加这些主题。用户以后亲自重新明确说明时，才会解除对应限制。
- `manual_notes` 可填写希望角色长期理解的简短补充。它不是角色身份或工具指令，不能改变固定人设、安全边界或权限。
- `extensions` 留给未来功能使用，当前不会作为可执行指令。

每次有效写入采用临时文件原子替换，并保存上一版 `.bak`。核心记忆还会在 `backups/` 中轮换保留最近 8 个版本。

固定角色人设仍来自 `characters/profiles/*.yaml`。记忆只是运行时补充，不会自动修改 YAML。
