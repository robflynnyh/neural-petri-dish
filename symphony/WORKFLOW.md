---
tracker:
  kind: linear
  project_slug: "neural-petri-dish-0253f8ae62a6"
  api_key: $LINEAR_API_KEY
  active_states:
    - Todo
    - In Progress
  terminal_states:
    - Closed
    - Cancelled
    - Canceled
    - Duplicate
    - Done
polling:
  interval_ms: 30000
workspace:
  root: /exp/exp4/acp21rjf/symphony-workspaces-neural-petri-dish
hooks:
  timeout_ms: 120000
  after_create: |
    if [ -f /exp/exp4/acp21rjf/neural-petri-dish/symphony/.env ]; then
      set -a
      . /exp/exp4/acp21rjf/neural-petri-dish/symphony/.env
      set +a
    fi
    git clone --depth 1 "$SOURCE_REPO_URL" .
    if [ -n "${SOURCE_REF:-}" ]; then
      git fetch --depth 1 origin "$SOURCE_REF"
      git checkout -B symphony-source FETCH_HEAD
    fi
agent:
  max_concurrent_agents: 1
  max_turns: 30
codex:
  command: /home/acp21rjf/.npm-global/bin/codex --config shell_environment_policy.inherit=all --config model_reasoning_effort=medium app-server
  approval_policy: never
  thread_sandbox: danger-full-access
  issue_label_overrides: true
  turn_sandbox_policy:
    type: dangerFullAccess
---

You are working on Linear issue {{ issue.identifier }} for the
neural-petri-dish repository.

Title: {{ issue.title }}
Current status: {{ issue.state }}
URL: {{ issue.url }}

Description:
{% if issue.description %}
{{ issue.description }}
{% else %}
No description provided.
{% endif %}

Required instruction index:
- Before planning or editing, read every file in `symphony/instructions/` in
  the order below.
- Treat those files as binding project instructions, not optional background.
- In your plan, state that you read them and note any instruction that directly
  affects the issue.

Read order:
1. `symphony/instructions/linear-context.md`
2. `symphony/instructions/repository.md`
3. `symphony/instructions/work-loop.md`
4. `symphony/instructions/experiment-execution.md`
5. `symphony/instructions/validation-and-handoff.md`

Instruction map:
- Linear issue context, comment rereads, and clarification handling:
  `symphony/instructions/linear-context.md`
- Repository layout, simulation rules, and artifact policy:
  `symphony/instructions/repository.md`
- Branch setup, planning, implementation style, and progress comments:
  `symphony/instructions/work-loop.md`
- GPU scheduler use, long renders, and experiment discipline:
  `symphony/instructions/experiment-execution.md`
- Validation, GitHub PR handoff, Linear review handoff, and final state rules:
  `symphony/instructions/validation-and-handoff.md`
