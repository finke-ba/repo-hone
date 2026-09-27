# RepoHone

**RepoHone learns from what goes wrong in your coding-agent sessions and turns it
into lasting improvements your project owns: a lint rule, a test, a stricter type
and, only as a last resort, one more line in `CLAUDE.md`.**

> Capture continuously. Analyze intentionally. Improve experimentally.

**Status: beta (0.4).** Works with Claude Code on macOS and Linux. Codex and Windows
are not supported yet. Commands and on-disk formats may still change between
releases.

## Why

Coding agents repeat the same mistakes. You correct one ("we never call the DB from
handlers here"), and the lesson dies with the session, or becomes another
instruction the agent may or may not follow next time. An instruction is the weakest
fix there is. A check that fails when the mistake happens again is the strongest.

## How it works

1. **It records your sessions quietly.** Your prompts, what the agent did (tool
   calls, including failed ones) and Git snapshots of the working tree at each turn,
   so it sees both the agent's changes and the edits you made afterwards.
   Everything stays on your machine, and recording makes no model calls.
2. **You choose what to analyse.** There are three starting points:
   - **A rule you state.** Say it while correcting the agent ("we always…"), and
     RepoHone records it together with the session it came from.
   - **Any session where something went wrong**, even if you never spelled out a
     rule. The model reads the session and the diffs and works out what happened:
     what you corrected, what caused it and what the project is missing. It may also
     answer that the evidence is too thin, or that nothing is worth changing.
   - **Your repository's history.** Instruction files, reverts and fix-up commits
     and, if you ask, pull-request review comments and CI failures become candidate
     rules.
3. **It proposes the strongest fix the project can own.** It first surveys what the
   repository already has (tests, linters, type checker, CI, hooks), then picks the
   most reliable mechanism that could carry the rule:

   ```
   structurally impossible → type/API → static rule → test → tooling → reviewer → instruction → docs
   ```

   A candidate check is validated in a separate checkout against the real states
   from your sessions: it has to flag the state with the mistake and accept the
   corrected one. An instruction, which cannot be run, is graded against past
   sessions instead.
4. **You decide.** Nothing changes without your approval, and every applied change
   can be rolled back.

Each diagnosis is fingerprinted, so a problem that keeps coming back is counted
across sessions (`repohone patterns`).

Three rules shape it:

- **Capture is automatic; analysis is explicit.** Nothing is analysed until you ask,
  and there is no backlog of suggestions waiting for you.
- **Analysis is not permission to change anything.** Every change needs your
  approval, and can be rolled back.
- **The project owns the fix.** Improvements live in your normal tooling
  (`make test`, linters, CI, hooks). RepoHone never sits in the execution path, and
  your project keeps working if you remove it.

## Usage

Requires Python ≥ 3.12, Git, Claude Code and [uv](https://docs.astral.sh/uv/) (or
pipx). On Linux, validating proposals also needs
[bubblewrap](https://github.com/containers/bubblewrap) (`bwrap`).

```bash
# 1. Once per machine: install the CLI and the Claude Code plugin (off everywhere)
uv tool install git+https://github.com/finke-ba/repo-hone
repohone install

# 2. Once per repository, per developer: turn it on here
repohone init
repohone doctor
```

`install` puts the hooks and the `repohone-rule` skill in `~/.claude/skills/repohone`.
`init` only enables them for this repo, via `.claude/settings.local.json`, and
creates `.repohone/profile.yaml`. Commit the profile so the team shares one capture
policy.

Then work as usual. When something is worth learning from:

| Step | How |
|---|---|
| Mark a rule (optional) | Say it while correcting the agent ("we always…"), or run `repohone rule "…"` |
| Diagnose | A marked rule: `repohone diagnose --candidate <rule-id>`. Any session: `repohone diagnose <session-id>` (`repohone list` shows them; `--turn N` points at the correction if you know it). Add `--show` to see exactly what would be sent, `--yes` to send it |
| Mine history | `repohone bootstrap [--include-remote]` |
| See what the repo already has | `repohone mechanisms [--probe]` |
| Propose and validate | `repohone propose <diagnosis-id> --yes` |
| Apply or undo | `repohone apply <proposal-id> --yes`, `repohone rollback <proposal-id>` |
| Explore | `repohone rules`, `patterns`, `diagnoses`, `proposals` |

`repohone deinit` turns it off in a repository, and `repohone uninstall` removes it
from the machine.

### From the chat

Once RepoHone is on, you can drive it from Claude Code's chat, in your own words
("analyse this session", "what has RepoHone found?", "apply it") or with a slash
command, one per CLI command (`/repohone:status`, `/repohone:propose`,
`/repohone:apply`, …). Ids are optional and RepoHone says which one it picked:
`/repohone:diagnose` with no arguments analyses a rule you marked, or else the
current session.

Anything that sends evidence to a model, runs the project's commands, reads GitHub,
changes your consent or changes files waits for your approval in Claude Code's own
prompt. In a terminal, `apply` and `rollback` only preview until you add `--yes`.

Analysis runs on the model the profile names (`reasoning_model`, Opus when it names
none). `--model` overrides it for one run; `repohone doctor` shows which model will be
used and why.

## Architecture

```
Claude Code session
   │ hooks: SessionStart · UserPromptSubmit · tool use · Stop · SessionEnd
   ▼
Adapter (thin, per agent) ──► RepoHone Core (Python)             
                                 ├─ capture     session records + Git snapshots
                                 ├─ storage     local SQLite (WAL) per checkout
                                 ├─ diagnosis   LLM, on request only
                                 ├─ mechanisms  what the repo already enforces
                                 ├─ proposal    validate in a detached checkout,
                                 │              apply with approval, roll back
                                 └─ bootstrap   rule candidates from repo history
```

- **Evidence stays local.** Records live in platform app-data storage, and
  snapshots live under `refs/repohone/` in your Git store.
- **Privacy** is set by the profile (`content_policy`, `tool_capture`,
  `reasoning_egress`). Content is redacted by default, and by default you approve
  each time evidence is sent to the model.
- **Validation never touches your working tree.** Candidate fixes are tested
  against real historical states in a separate checkout.

## What's coming

- **Spotting corrections on its own.** RepoHone will recognise a correction without
  you marking it, and find candidate rules in mistakes that keep coming back. You
  see them when you ask. Background work stays local and mechanical, with no model
  calls.
- **Measuring whether a fix helped.** Each applied change will be watched for fewer
  repeat corrections, and for its cost in context and friction.
- **Removing what doesn't work.** A change that isn't helping, or has been superseded,
  will come back to you as a proposal to remove or revise it. RepoHone will also say
  when there is nothing left worth changing, and stop proposing.
- **More places to run it.** Codex support, Windows, installing from the chat as a
  Claude Code plugin, and an offer to turn RepoHone on the first time you open a new
  repository.

Not planned for now: several developers sharing one checkout.

## Development

```bash
git clone https://github.com/finke-ba/repo-hone && cd repo-hone
make check     # compile, lint (ruff), type-check (mypy) and test
make install   # install your checkout as the repohone CLI, editable
```

`make help` lists every target. The suite needs Git ≥ 2.31, must not run as root,
and on Linux needs `bwrap`.

## License

[Apache 2.0](LICENSE)
