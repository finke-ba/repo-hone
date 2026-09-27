---
name: diagnose
description: Analyse why a session went wrong, or a project rule you marked, with the analysis model
disable-model-invocation: true
argument-hint: "[session id | rule id]"
allowed-tools: Bash({repohone} diagnose --show:*)
---

# Diagnose

What the developer typed after the command: `$ARGUMENTS`

1. Run `{repohone} diagnose --show`, adding what they named: a rule id (it starts
   with `rule_`) as `--candidate <id>`, a session id as it is. RepoHone says which
   target it picked and summarises what it would send.
2. Show that summary as printed.
3. Run `{repohone} diagnose <session id> --yes`, or for a rule
   `{repohone} diagnose --candidate <rule id> --yes`, naming the target the
   preview picked. Give the Bash call a timeout of 600000 ms.
4. Show the diagnosis as printed. Then one line: `/repohone:propose` designs an
   improvement for it.

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
