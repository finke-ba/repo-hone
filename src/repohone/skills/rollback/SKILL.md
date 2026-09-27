---
name: rollback
description: Undo an improvement RepoHone applied
disable-model-invocation: true
argument-hint: "[proposal id]"
---

# Roll back

What the developer typed after the command: `$ARGUMENTS`

1. Run `{repohone} rollback`, adding the proposal id if they named one. RepoHone
   says which proposal it picked and which files it would restore; it exits 3
   because nothing is undone yet.
2. Show that plan as printed.
3. Run `{repohone} rollback <proposal id> --yes`, naming the proposal shown.
4. Show the result as printed.

**Never restore the files any other way**: RepoHone checks each one still holds
what it applied, so a later edit is not silently discarded.

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
