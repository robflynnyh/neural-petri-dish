#!/usr/bin/env python3
"""Post queued-experiment completion back to Linear.

This helper is intended for detached experiment wrappers. It uses the Linear
GraphQL API directly because long-running shell processes cannot call Codex
tools such as linear_graphql after the agent has exited.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional


LINEAR_API_URL = "https://api.linear.app/graphql"
DEFAULT_MAX_COMMENT_CHARS = 10000
DEFAULT_MAX_LOG_CHARS = 6000


class LinearError(RuntimeError):
    pass


def graphql(api_key: str, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    payload = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    request = urllib.request.Request(
        LINEAR_API_URL,
        data=payload,
        headers={
            "Authorization": api_key,
            "Content-Type": "application/json",
            "User-Agent": "neural-petri-dish-experiment-callback",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise LinearError(f"Linear API HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise LinearError(f"Linear API request failed: {error}") from error

    decoded = json.loads(body)
    if decoded.get("errors"):
        raise LinearError(json.dumps(decoded["errors"], indent=2))
    return decoded["data"]


def fetch_issue(api_key: str, issue: str) -> Dict[str, Any]:
    query = """
    query IssueForExperimentCallback($id: String!) {
      issue(id: $id) {
        id
        identifier
        title
        url
        state {
          name
        }
        team {
          states(first: 100) {
            nodes {
              id
              name
              type
            }
          }
        }
      }
    }
    """
    data = graphql(api_key, query, {"id": issue})
    found = data.get("issue")
    if not found:
        raise LinearError(f"Linear issue not found: {issue}")
    return found


def state_id(issue: Dict[str, Any], target_state: str) -> str:
    states = issue["team"]["states"]["nodes"]
    for state in states:
        if state["name"].lower() == target_state.lower():
            return state["id"]
    available = ", ".join(sorted(state["name"] for state in states))
    raise LinearError(f"State {target_state!r} not found. Available states: {available}")


def post_comment(api_key: str, issue_id: str, body: str) -> None:
    mutation = """
    mutation ExperimentCallbackComment($input: CommentCreateInput!) {
      commentCreate(input: $input) {
        success
      }
    }
    """
    data = graphql(api_key, mutation, {"input": {"issueId": issue_id, "body": body}})
    if not data["commentCreate"]["success"]:
        raise LinearError("Linear commentCreate returned success=false")


def move_issue(api_key: str, issue_id: str, target_state_id: str) -> None:
    mutation = """
    mutation ExperimentCallbackState($id: String!, $input: IssueUpdateInput!) {
      issueUpdate(id: $id, input: $input) {
        success
      }
    }
    """
    data = graphql(api_key, mutation, {"id": issue_id, "input": {"stateId": target_state_id}})
    if not data["issueUpdate"]["success"]:
        raise LinearError("Linear issueUpdate returned success=false")


def cap_text(text: str, max_chars: int, label: str) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = f"\n\n[{label} truncated to {max_chars} characters]\n"
    keep = max_chars - len(marker)
    if keep <= 0:
        return marker[-max_chars:]
    return marker + text[-keep:]


def cap_comment(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = f"\n\n[Linear callback comment truncated to {max_chars} characters]"
    keep = max_chars - len(marker)
    if keep <= 0:
        return marker[:max_chars]
    return text[:keep].rstrip() + marker


def tail(path: Optional[str], lines: int, max_chars: int) -> str:
    if not path:
        return ""
    file_path = Path(path)
    if not file_path.exists():
        return f"Log file does not exist: `{path}`"
    try:
        content = file_path.read_text(errors="replace").splitlines()
    except OSError as error:
        return f"Could not read log file `{path}`: {error}"
    selected = content[-lines:]
    return cap_text("\n".join(selected), max_chars, "log tail")


def existing_path_status(path: Optional[str]) -> str:
    if not path:
        return "not provided"
    return "exists" if Path(path).exists() else "missing"


def build_comment(args: argparse.Namespace, issue: Dict[str, Any]) -> str:
    status_code = int(args.status_code)
    outcome = "succeeded" if status_code == 0 else "failed"
    branch = args.branch or os.environ.get("GIT_BRANCH") or "not provided"
    commit = args.commit or os.environ.get("GIT_COMMIT") or "not provided"
    queued_command = args.queued_command or os.environ.get("QUEUED_COMMAND") or "not provided"
    screen_name = args.screen_name or os.environ.get("SCREEN_NAME") or "not provided"
    runner_label = args.runner_label or os.environ.get("RUNNER_LABEL")
    if not runner_label and screen_name != "not provided":
        runner_label = f"screen:{screen_name}"
    if not runner_label:
        runner_label = "not provided"
    log_tail = tail(args.log, args.tail_lines, args.max_log_chars)

    parts = [
        f"Queued experiment {outcome}.",
        "",
        f"- Exit status: `{status_code}`",
        f"- Issue: `{issue['identifier']}`",
        f"- Runner: `{runner_label}`",
        f"- Log path: `{args.log or 'not provided'}`",
        f"- Results path: `{args.results or 'not provided'}` ({existing_path_status(args.results)})",
        f"- Branch: `{branch}`",
        f"- Commit: `{commit}`",
        f"- Queued command: `{queued_command}`",
        f"- Callback target state: `{args.target_state}`",
    ]
    if args.note:
        parts.extend(["", args.note])
    if log_tail:
        parts.extend(["", f"Last {args.tail_lines} log lines:", "```text", log_tail, "```"])
    return cap_comment("\n".join(parts), args.max_comment_chars)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue", required=True, help="Linear issue id or identifier, e.g. ROB-44")
    parser.add_argument("--status-code", required=True, type=int, help="Experiment process exit status")
    parser.add_argument("--log", help="Experiment log path")
    parser.add_argument("--results", help="Expected result path")
    parser.add_argument("--screen-name", help="Detached screen session name")
    parser.add_argument("--runner-label", help="Execution handle, e.g. screen:<name> or slurm:<job_id>")
    parser.add_argument("--queued-command", help="Exact queued command or wrapper path")
    parser.add_argument("--branch", help="Git branch used for the run")
    parser.add_argument("--commit", help="Git commit used for the run")
    parser.add_argument("--target-state", default="Todo", help="Linear state to move the issue to")
    parser.add_argument("--note", help="Extra Markdown note to include in the Linear comment")
    parser.add_argument("--tail-lines", type=int, default=80, help="Number of log lines to include")
    parser.add_argument(
        "--max-log-chars",
        type=int,
        default=DEFAULT_MAX_LOG_CHARS,
        help="Maximum characters of log tail to include in the comment",
    )
    parser.add_argument(
        "--max-comment-chars",
        type=int,
        default=DEFAULT_MAX_COMMENT_CHARS,
        help="Maximum characters to post to Linear",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the comment and skip Linear mutations")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate Linear API access, issue lookup, and target state without mutating Linear",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("LINEAR_API_KEY")
    if not api_key and not args.dry_run:
        raise LinearError("LINEAR_API_KEY is required unless --dry-run is used")

    if args.dry_run:
        issue = {
            "id": args.issue,
            "identifier": args.issue,
            "title": "",
            "url": "",
            "state": {"name": ""},
            "team": {"states": {"nodes": [{"id": args.target_state, "name": args.target_state, "type": ""}]}},
        }
        print(build_comment(args, issue))
        return 0

    issue = fetch_issue(api_key, args.issue)
    target_state_id = state_id(issue, args.target_state)
    if args.check_only:
        print(f"Linear issue {issue['identifier']} can be moved to {args.target_state} ({target_state_id}).")
        return 0
    comment = build_comment(args, issue)
    post_comment(api_key, issue["id"], comment)
    move_issue(api_key, issue["id"], target_state_id)
    print(f"Posted completion comment and moved {issue['identifier']} to {args.target_state}.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LinearError as error:
        print(f"linear_experiment_callback: {error}", file=sys.stderr)
        raise SystemExit(1)
