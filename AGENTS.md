## Agent skills

### Issue tracker

Issues live in GitHub Issues (`LostofMoon/qwevl_algo`). See `docs/agents/issue-tracker.md`.

### Triage labels

Default label vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context repo — one `CONTEXT.md` + `docs/adr/` at the root. See `docs/agents/domain.md`.

## Project context

This is a research repo modifying the Qwen3-VL vision-language model architecture.

- **Active development directory**: `qwen3vl_improved/` — modified version of `transformers` 4.57.0 Qwen3-VL source with absolute imports and algorithm changes
- **Reference directory**: `qwen3vl_original/` — read-only copy of the installed transformers source, do not modify
- **Entry point**: `run_example.py`

Code runs on a remote GPU server (H20) via SSH. Do not run git commands unless explicitly asked.
