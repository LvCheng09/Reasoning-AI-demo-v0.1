"""CLI entrypoint for the main-part reasoning system."""

from __future__ import annotations

import json
from pathlib import Path

from .model_launcher import MainPartModelLauncher
from .runtime_status import execution_outcome, exit_code


def _load_json_arg(raw: str, path: str) -> dict | None:
    if raw and path:
        raise ValueError("Use only one of --frontend-state-json or --frontend-state-file.")
    if raw:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("--frontend-state-json must decode to a JSON object.")
        return parsed
    if path:
        parsed = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("--frontend-state-file must contain a JSON object.")
        return parsed
    return None


def _runtime_completed(result: dict) -> bool:
    return execution_outcome(result) in {"solved", "blocked"}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the main reasoning model launcher.")
    parser.add_argument("--input", required=True, help="Single raw input for the frontend LLM.")
    parser.add_argument("--frontend-state-json", default="", help="Optional frontend state JSON for no-LLM runs.")
    parser.add_argument("--frontend-state-file", default="", help="Optional path to frontend state JSON for no-LLM runs.")
    parser.add_argument("--session-id", default="main_runtime", help="Runtime session id.")
    parser.add_argument("--no-auto-run", action="store_true", help="Only write frontend output into working memory.")
    parser.add_argument("--pretty", action="store_true", help="Print indented JSON.")
    args = parser.parse_args()

    payload: dict = {"input": args.input}
    frontend_state = _load_json_arg(args.frontend_state_json, args.frontend_state_file)
    launcher = MainPartModelLauncher(session_id=args.session_id, auto_run=not args.no_auto_run)
    result = (launcher.submit_event_graph(payload, frontend_state)
              if frontend_state is not None else launcher.submit_input(payload))
    print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None, default=str))
    if not args.no_auto_run:
        raise SystemExit(exit_code(execution_outcome(result)))


if __name__ == "__main__":
    main()
