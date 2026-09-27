---
name: init
description: Turn RepoHone on for you in this repository by accepting its profile
disable-model-invocation: true
allowed-tools: Bash({repohone} status)
---

# Turn RepoHone on here

1. Run `{repohone} status`. If capture is already on, say so and stop.
2. Show the project's profile, `.repohone/profile.yaml`, as it is; if there is
   none, say RepoHone will create the default one, which the team then commits.
   Say in one line what accepting means: RepoHone captures your sessions here, on
   this machine, and sends nothing anywhere until you ask for an analysis.
3. Run `{repohone} init`.
4. Show the result as printed.

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
