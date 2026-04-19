# Claude Code setup (macOS, zsh)

This document describes a **shell-based** workflow: `~/.zshrc` + `~/.claude-keys.zsh`. It is not tied to a single git repo.

## Summary

| Goal | Command / tool |
|------|----------------|
| Pick NVIDIA / OpenRouter / Ollama and run **Claude Code** directly | `claude-free` |
| Ollama only (local) | `claude-local` |
| **NVIDIA NIM with Claude Code** (recommended) | **Claudish** — `claudish-nvidia` / `claudish-nvidia-free` |

## Why Claudish for NVIDIA?

NVIDIA [integrate.api.nvidia.com](https://integrate.api.nvidia.com) exposes an **OpenAI-compatible** `POST /v1/chat/completions` API. **Claude Code** normally speaks an **Anthropic-compatible** protocol. Pointing `ANTHROPIC_BASE_URL` straight at NVIDIA often fails even when **curl** to the same endpoint works.

**Claudish** proxies Claude Code through a local Anthropic-compatible server and translates to OpenAI (and other providers). For NVIDIA, use the **`oai@`** route with:

- `OPENAI_API_KEY` = your `nvapi-...` key  
- `OPENAI_BASE_URL` = `https://integrate.api.nvidia.com/v1`  
- `claudish --model oai@<nVIDIA-model-id>`

See [Claudish README](https://github.com/MadAppGang/claudish) (`OPENAI_BASE_URL`, `oai@`).

## Prerequisites

- [Node.js](https://nodejs.org) (for `npm`)
- [Claude Code](https://www.npmjs.com/package/@anthropic-ai/claude-code): `npm install -g @anthropic-ai/claude-code`
- **Claudish** (for NVIDIA via OpenAI-compatible API):  
  `npm install -g claudish`  
  Claudish expects the **[Bun](https://bun.sh)** runtime:  
  `curl -fsSL https://bun.sh/install | bash`  
  (Adds `~/.bun/bin` to your `PATH`.)

> **Homebrew** install of Claudish may require up-to-date Xcode Command Line Tools. If `brew install claudish` fails, use `npm` + Bun as above.

## Files

| File | Role |
|------|------|
| `~/.zshrc` | `claude-free`, `claude-local`, `claudish-nvidia`, model lists |
| `~/.claude-keys.zsh` | `_NVIDIA_KEYS`, `_OPENROUTER_KEYS` (keep private; do not commit) |

Set Ollama URL if not default `11434`:

```bash
export _CLAUDE_OLLAMA_URL='http://localhost:11444'
```

(Or add it before sourcing `~/.claude-keys.zsh` in `~/.zshrc`.)

## NVIDIA models (edit in `~/.zshrc`)

Array `_NVIDIA_MODELS` — use the exact model strings from NVIDIA’s “Copy code” for each model.

Examples:

- `deepseek-ai/deepseek-v3.2`
- `moonshotai/kimi-k2-instruct`
- `z-ai/glm4.7`

## Daily usage

### 1) NVIDIA via Claudish (recommended)

Single model (default: DeepSeek):

```bash
claudish-nvidia
```

Explicit model:

```bash
claudish-nvidia moonshotai/kimi-k2-instruct
```

Interactive menu (NVIDIA list only):

```bash
claudish-nvidia-free
```

These commands **rotate** through `_NVIDIA_KEYS` in `~/.claude-keys.zsh`.

### Claudish UI vs plain Claude Code

By default, Claudish can wrap interactive sessions in **mtm** (its multiplexer), which makes the terminal look different from running `claude` directly. To match the **normal Claude Code TUI**, use diagnostic mode **off** (no mtm; Claude runs in a plain process):

- **One-off:** `CLAUDISH_DIAG_MODE=off claudish --model 'oai@…'`
- **Global default:** add `"diagMode": "off"` to `~/.claudish/config.json`
- The `claudish-nvidia` helper in `~/.zshrc` sets `CLAUDISH_DIAG_MODE=off` by default (override with `CLAUDISH_DIAG_MODE=auto` if you want the old behavior).

### 2) Multi-provider picker (Claude Code only — not for NVIDIA reliability)

```bash
claude-free
```

Use **OpenRouter** or **Ollama** here; for **NVIDIA**, prefer `claudish-nvidia` above.

### 3) Local Ollama only

```bash
claude-local
```

## Troubleshooting

### “Detected a custom API key” (Claudish / Claude)

Claudish uses `ANTHROPIC_API_KEY=sk-ant-api03-placeholder` to suppress Claude Code login prompts; real traffic for NVIDIA is `OPENAI_API_KEY` + `OPENAI_BASE_URL` → follow prompts as needed.

### Claudish: `claudish requires the Bun runtime`

Install Bun and ensure `~/.bun/bin` is on `PATH` (the installer usually appends to `~/.zshrc`).

### Verify NVIDIA API (outside Claude)

```bash
curl -sS https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer nvapi-..." \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-ai/deepseek-v3.2","messages":[{"role":"user","content":"hi"}],"max_tokens":50}'
```

Replace `nvapi-...` with a real key from your key file.

## Security

- Never paste API keys into chats or screenshots.
- Rotate keys if exposed.
- Keep `~/.claude-keys.zsh` out of git and cloud-synced public folders if possible.

## Next steps (optional)

- **Automatic model selection** by prompt (e.g. Claudish profiles, routing rules, or multi-model mesh) — see [Claudish docs](https://github.com/MadAppGang/claudish) and [claudish.com](https://claudish.com).
