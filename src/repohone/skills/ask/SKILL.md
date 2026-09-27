---
name: ask
description: Use when the developer asks RepoHone something or asks it to act — what it captured or found, recurring patterns, diagnoses, proposals — or to analyse a session, apply or undo an improvement. Not for stating a project rule; that is repohone-rule.
allowed-tools: Bash({repohone} status), Bash({repohone} doctor), Bash({repohone} diagnose --show:*), Bash({repohone} propose --show:*), Bash({repohone} deinit --dry-run), Bash({repohone} list --json), Bash({repohone} rules --json), Bash({repohone} diagnoses --json), Bash({repohone} proposals --json), Bash({repohone} patterns)
---

# Answering and acting for RepoHone

RepoHone runs as `{repohone}`. Pick what the developer asked for in their own
words; ids are optional everywhere, because RepoHone picks the default target and
says which.

**Questions** — run and summarise, claiming nothing the output does not say:

| they ask about | run |
|---|---|
| whether RepoHone is on here | `{repohone} status` |
| installation problems | `{repohone} doctor` |
| captured sessions | `{repohone} list --json` |
| marked rules | `{repohone} rules --json` |
| recurring problems | `{repohone} patterns` |
| diagnoses | `{repohone} diagnoses --json` |
| proposals | `{repohone} proposals --json` |

**Actions** — only when the developer asked for this one in this message;
otherwise offer it in one line and name its slash command.

| they ask to | preview, shown as printed | then |
|---|---|---|
| analyse a session or rule (`/repohone:diagnose`) | `{repohone} diagnose --show` | `{repohone} diagnose <session id> --yes` or `{repohone} diagnose --candidate <rule id> --yes`, Bash timeout 600000 ms |
| design an improvement (`/repohone:propose`) | `{repohone} propose --show` | `{repohone} propose <diagnosis id> --yes`, in the background |
| apply it (`/repohone:apply`) | `{repohone} apply` | `{repohone} apply <proposal id> --yes` |
| undo it (`/repohone:rollback`) | `{repohone} rollback` | `{repohone} rollback <proposal id> --yes` |
| turn RepoHone on here (`/repohone:init`) | `{repohone} status` and the profile | `{repohone} init` |
| turn it off here (`/repohone:deinit`) | `{repohone} deinit --dry-run` | `{repohone} deinit --yes` |

The `then` command names the target its preview picked. Apply and roll back only
through RepoHone, never by editing the files yourself: that skips its backups and
undo record. Anything else — measuring checks, mining history, uninstalling —
point the developer to `/repohone:mechanisms`, `/repohone:bootstrap` or
`/repohone:uninstall`.

## Rules

- Run RepoHone only as written here, ids spelled out literally: no shell
  variables, or the approval will not match.
- Do not ask in the chat before the command that needs approval. Claude Code
  shows the developer an approval prompt; that is the one approval. If they
  decline, say so and stop: never retry it or reach the same result another way.
- Show what the developer approves exactly as RepoHone printed it, in a code
  block. Summarise anything else in a sentence or two, claiming nothing the output
  does not say.
- If RepoHone picks no target, relay what it printed and ask one question.
