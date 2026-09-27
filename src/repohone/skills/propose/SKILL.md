---
name: propose
description: Design and validate an improvement for a diagnosis, with the analysis model
disable-model-invocation: true
argument-hint: "[diagnosis id]"
allowed-tools: Bash({repohone} propose --show:*)
---

# Propose

What the developer typed after the command: `$ARGUMENTS`

1. Run `{repohone} propose --show`, adding the diagnosis id if they named one.
   RepoHone says which diagnosis it picked and summarises what it would send.
2. Show that summary as printed.
3. Run `{repohone} propose <diagnosis id> --yes` in the background
   (`run_in_background`), naming the diagnosis the preview picked: validation runs
   the project's own checks and can take several minutes. Wait for it to finish.
4. Show the proposal as printed, including the exact change. Then one line:
   `/repohone:apply` applies it.

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
