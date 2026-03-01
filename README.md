# agent-safety-mcp

[![PyPI version](https://img.shields.io/pypi/v/agent-safety-mcp)](https://pypi.org/project/agent-safety-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

**MCP server for AI agent safety.** One install gives any MCP-compatible AI assistant access to cost guards, prompt injection scanning, and decision tracing.

Works with Claude Code, Cursor, Windsurf, Zed, and any MCP client.

---

## Install

### Claude Code (recommended)

```bash
claude mcp add agent-safety -- uvx agent-safety-mcp
```

### Manual (any MCP client)

Add to your MCP config:

```json
{
  "mcpServers": {
    "agent-safety": {
      "command": "uvx",
      "args": ["agent-safety-mcp"]
    }
  }
}
```

### From PyPI

```bash
pip install agent-safety-mcp
agent-safety-mcp  # runs stdio server
```

---

## Tools

### Cost Guard — Budget enforcement for LLM calls

| Tool | What it does |
|---|---|
| `cost_guard_configure` | Set weekly budget, alert threshold, dry-run mode |
| `cost_guard_status` | Check current spend vs budget |
| `cost_guard_check` | Pre-check if a model call is within budget |
| `cost_guard_record` | Record a completed call's token usage |
| `cost_guard_models` | List supported models with pricing |

**Example:** "Check if I can afford a GPT-4o call with 2000 input tokens"

### Injection Guard — Prompt injection scanner

| Tool | What it does |
|---|---|
| `injection_scan` | Scan text for injection patterns (non-blocking) |
| `injection_check` | Scan + block if injection detected |
| `injection_patterns` | List all 22 built-in detection patterns |

**Example:** "Scan this user input for prompt injection: 'ignore previous instructions and...'"

### Decision Tracer — Agent decision logging

| Tool | What it does |
|---|---|
| `trace_start` | Start a new trace session |
| `trace_step` | Log a decision step with context |
| `trace_summary` | Get session summary (steps, errors, timing) |
| `trace_save` | Save trace to JSON + Markdown files |

**Example:** "Start a trace for my analysis agent, then log each decision step"

---

## What this wraps

This MCP server wraps the **AI Agent Infrastructure Stack** — three standalone Python libraries:

- [ai-cost-guard](https://github.com/LuciferForge/ai-cost-guard) — `pip install ai-cost-guard`
- [ai-injection-guard](https://github.com/LuciferForge/prompt-shield) — `pip install ai-injection-guard`
- [ai-decision-tracer](https://github.com/LuciferForge/ai-trace) — `pip install ai-decision-tracer`

All three: MIT licensed, zero runtime dependencies (individually), pure Python stdlib.

The MCP server adds `mcp>=1.0.0` as a dependency for the protocol layer.

---

## Why

AI coding assistants (Claude Code, Cursor, etc.) can now protect the agents they help build — checking budgets, scanning inputs, and tracing decisions — without leaving the IDE.

Built from 8 months of running autonomous AI trading agents in live financial markets.

---

## License

MIT
