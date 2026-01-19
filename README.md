# Claude OpenAI Tools Bridge Pipe (OpenWebUI)

作者 / Author: **JiangNanGenius**  
仓库 / Repo: https://github.com/JiangNanGenius/  
版本 / Version: **6.3.1**

---

## 中文简介

这是一个用于 **OpenWebUI Pipelines** 的桥接管道：

- 把 **Anthropic Messages API（Claude）** 的响应转换为 **OpenAI ChatCompletions** 兼容格式（`tools` / `tool_calls` / `role="tool"`），让 OpenWebUI 能用 **原生工具 UI** 执行并展示工具调用。
- 同时支持 **Claude Extended Thinking**（带 `signature` 的 signed thinking），并在工具续轮时自动回放签名 thinking，避免常见报错：
  - `Expected thinking or redacted_thinking, but found tool_use`
- 思考显示已“写死”为 **OpenWebUI 原生 `reasoning_content`**：
  - 不再用 `<think>...</think>` 标签拼接（更稳定，减少折叠/串流异常）。
  - **只要该回合会调用工具**，Claude 输出的普通文本也会被强制归入 `reasoning_content`（显示在“思考”折叠块里），避免在正文里混出“步骤/分析”。

> 你反馈的现象「第一次思考正常，调用完工具第二次思考变正文」通常是因为 UI 只识别第一段 `<think>` 或对标签解析不稳定。这个版本改为走 `reasoning_content`，并把工具回合的文本统一收进思考块。

---

## English

This is an **OpenWebUI Pipelines** bridge that:

- Converts **Anthropic Messages API (Claude)** responses into **OpenAI ChatCompletions-compatible** output (`tools` / `tool_calls` / `role="tool"`), so OpenWebUI can run tools with its **native tool UI**.
- Supports **Claude Extended Thinking** (signed thinking with `signature`) and replays the signed thinking block before `tool_use` on tool-continuation turns to avoid:
  - `Expected thinking or redacted_thinking, but found tool_use`
- Thinking UI is hard-coded to **OpenWebUI-native `reasoning_content`**:
  - No `<think>...</think>` tag hacks (more stable).
  - If the assistant turn uses tools, any normal text is routed into `reasoning_content` so only the final answer appears as normal content.

---

## 文件 / Files

- `openwebui_pipe_claude_openai_tools.py`  
  Pipeline 主文件 / Main pipeline file.

- `README.md`  
  使用说明 / Documentation.

---

## 安装 / Install

1. 把 `openwebui_pipe_claude_openai_tools.py` 放入 OpenWebUI 的 pipelines 目录  
   （不同部署目录名可能不同：`pipelines/`、`custom_pipelines/` 等）。

2. 重启 OpenWebUI。

3. 在模型/Provider 列表中，选择本 Pipe 暴露的模型（前缀一般为 `claude-openai-tools/`）。

---

## 环境变量 / Environment Variables

### 必填 / Required

```bash
export ANTHROPIC_API_KEY="YOUR_KEY"
```

### 可选：上游地址 / Optional: Upstream Base URL

```bash
# Official
export ANTHROPIC_BASE_URL="https://api.anthropic.com"

# Or an Anthropic-compatible proxy base URL
# export ANTHROPIC_BASE_URL="https://api.openai-proxy.org/anthropic"
```

### 其他常用项 / Other knobs (optional)

```bash
# HTTP timeouts
export ANTHROPIC_CONNECT_TIMEOUT="10"
export ANTHROPIC_READ_TIMEOUT="300"

# Tool-call streaming mode: single (default) / delta
export ANTHROPIC_TOOL_CALL_STREAM_MODE="single"

# Enable signed thinking replay when thinking+tools are used (default true)
export ANTHROPIC_ALLOW_THINKING_WITH_TOOLS="true"

# Logging
export ANTHROPIC_LOG_ENABLED="false"
export ANTHROPIC_LOG_LEVEL="WARNING"
```

---

## 重要说明 / Notes

### 1) 为什么“工具回合的文字”会进思考块？

为了满足你的需求（工具回合不在正文里混出分析/步骤），本 Pipe 做了一个稳定策略：

- 只要该 assistant 回合出现 `tool_use`（最终 `finish_reason=tool_calls`）
- 那这一回合里 Claude 输出的普通文本，会被当作 `reasoning_content` 发送给 OpenWebUI

这样 OpenWebUI 会把它放在“思考”折叠块里，而不是正文。

### 2) 可能看到“非工具回合”最后一次性输出

由于需要先判断该回合是否会调用工具，本 Pipe 会先缓存正文文本：

- 如果最终没有调用工具：在 `MESSAGE_STOP` 时一次性把缓存的正文发出来
- 如果调用工具：缓存内容会进入思考块

这是在“不让工具回合漏正文”前提下的取舍。

---

## 作者 / Author

- **JiangNanGenius**  
  https://github.com/JiangNanGenius/
