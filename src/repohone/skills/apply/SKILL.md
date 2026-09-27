---
name: apply
description: Apply a validated RepoHone proposal to your working tree
disable-model-invocation: true
argument-hint: "[proposal id]"
---

# Apply

What the developer typed after the command: `$ARGUMENTS`

1. Run `{repohone} apply`, adding the proposal id if they named one. RepoHone
   says which proposal it picked, what happened, and the exact change; it exits 3
   because nothing is applied yet.
2. Show all of it as printed.
3. Run `{repohone} apply <proposal id> --yes`, naming the proposal shown.
4. Show the result as printed. Then one line: `/repohone:rollback` undoes it.

**Never make the change any other way**, not even to fix a conflict: an edit you
make yourself skips RepoHone's backups and its undo record.

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
