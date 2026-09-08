from __future__ import annotations

import json
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Any

from .harbor_tasks import EDIT_TASKS, FUNCTION_TASKS, REPAIR_TASKS, STATEFUL_TASKS

# Harbor v0.22.0, src/harbor/agents/terminus_2/templates/terminus-json-plain.txt.
# The braces are doubled because this is formatted with the str.format().
TERMINUS_JSON_TEMPLATE = r'''You are an AI assistant tasked with solving command-line tasks in a Linux environment. You will be given a task description and the output from previously executed commands. Your goal is to solve the task by providing batches of shell commands.

Format your response as JSON with the following structure:

{{
  "analysis": "Analyze the current state based on the terminal output provided. What do you see? What has been accomplished? What still needs to be done?",
  "plan": "Describe your plan for the next steps. What commands will you run and why? Be specific about what you expect each command to accomplish.",
  "commands": [
    {{
      "keystrokes": "ls -la\n",
      "duration": 0.1
    }}
  ],
  "task_complete": true
}}

Required fields:
- "analysis": Your analysis of the current situation
- "plan": Your plan for the next steps
- "commands": Array of command objects to execute

Optional fields:
- "task_complete": Boolean indicating if the task is complete (defaults to false if not present)

The text inside "keystrokes" is used verbatim. End every command with a newline. The JSON must be valid. Do not put text before or after the JSON.

Task Description:
{instruction}

Current terminal state:
{terminal_state}
'''

CONFIRMATION = (
    "Are you sure you want to mark the task as complete? This will trigger your solution to be graded and you "
    'will not be able to make any further corrections. If so, include "task_complete": true in your JSON '
    "response again."
)


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _assistant(
    analysis: str,
    plan: str,
    commands: Iterable[tuple[str, float]] = (),
    *,
    task_complete: bool = False,
) -> dict[str, str]:
    payload = {
        "analysis": analysis,
        "plan": plan,
        "commands": [{"keystrokes": command, "duration": duration} for command, duration in commands],
        "task_complete": task_complete,
    }
    return {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}


def _user(content: str) -> dict[str, str]:
    return {"role": "user", "content": content}


def _initial(instruction: str) -> dict[str, str]:
    terminal_state = "Current Terminal Screen:\nApptainer>"
    return _user(TERMINUS_JSON_TEMPLATE.format(instruction=instruction, terminal_state=terminal_state))


def _terminal(command: str, output: str = "", *, confirmation: bool = False) -> dict[str, str]:
    rendered = f"Current terminal state:\nNew Terminal Output:\nApptainer> {command.rstrip()}\n{output}Apptainer>"
    if confirmation:
        rendered += f"\n\n{CONFIRMATION}"
    return _user(rendered)


def _confirm() -> dict[str, str]:
    return _assistant(
        "The requested change has been executed and checked.",
        "Confirm completion without running another command.",
        task_complete=True,
    )


def _record(task_name: str, category: str, instruction: str, messages: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "messages": messages,
        "metadata": {
            "task_name": f"openeurollm/{task_name}",
            "category": category,
            "language": "en",
            "license": "Apache-2.0",
            "source": "project-authored Harbor oracle",
            "protocol": "harbor-v0.22.0/terminus-2-json",
            "expected_reward": 1.0,
            "disposition": "sft-bridge-only",
            "not_for_rl_or_evaluation": True,
        },
    }


def _function_records() -> list[dict[str, Any]]:
    records = []
    for name, function, arguments in FUNCTION_TASKS:
        task_name = f"function-{name}"
        request = {"name": function, "arguments": arguments}
        compact = json.dumps(request, separators=(",", ":"))
        instruction = (
            f"Use `python3 tool_api.py '<JSON call>'` exactly once. Send tool={function} with arguments="
            f"{json.dumps(arguments, sort_keys=True)}. Do not edit audit or result files directly."
        )
        command = f"python3 tool_api.py '{compact}'\n"
        result = {
            "convert_temperature": arguments["value"] * 9 / 5 + 32 if function == "convert_temperature" else None,
            "scale_recipe": arguments["servings_to"] / arguments["servings_from"] if function == "scale_recipe" else None,
            "lookup_timezone": "Europe/Helsinki" if function == "lookup_timezone" else None,
            "compute_vat": arguments["net"] * 0.25 if function == "compute_vat" else None,
        }[function]
        output = json.dumps({"result": result}, sort_keys=True) + "\n"
        messages = [
            _initial(instruction),
            _assistant("The task supplies an exact deterministic tool call.", "Execute it once, then submit.", [(command, 0.1)], task_complete=True),
            _terminal(command, output, confirmation=True),
            _confirm(),
        ]
        records.append(_record(task_name, "function-calling", instruction, messages))
    return records


def _stateful_records() -> list[dict[str, Any]]:
    records = []
    for name, sku, quantity, delta in STATEFUL_TASKS:
        task_name = f"stateful-{name}"
        instruction = (
            f"Use `python3 inventory_tool.py get {sku}` first. Then update {sku} by {delta:+d} with the same "
            "tool. Do not directly edit the inventory or audit files."
        )
        get_command = f"python3 inventory_tool.py get {sku}\n"
        update_command = f"python3 inventory_tool.py update {sku} {delta}\n"
        messages = [
            _initial(instruction),
            _assistant("The current stock must be observed before mutation.", "Read the requested SKU.", [(get_command, 0.1)]),
            _terminal(get_command, f"{quantity}\n"),
            _assistant(
                f"The observed quantity is {quantity}; applying {delta:+d} gives {quantity + delta}.",
                "Run the required state-changing tool call and submit.",
                [(update_command, 0.1)],
                task_complete=True,
            ),
            _terminal(update_command, f"{quantity + delta}\n", confirmation=True),
            _confirm(),
        ]
        records.append(_record(task_name, "stateful-tools", instruction, messages))
    return records


def _edit_records() -> list[dict[str, Any]]:
    records = []
    for name, initial, expected in EDIT_TASKS:
        task_name = f"terminal-{name}"
        changed = next(key for key in expected if expected[key] != initial.get(key))
        instruction = f"Edit `config.json`: set `{changed}` to `{expected[changed]}` and preserve every other value."
        read_command = "cat config.json\n"
        edit_command = (
            "python3 -c \"import json; p='config.json'; d=json.load(open(p)); "
            f"d[{changed!r}]={expected[changed]!r}; "
            "open(p,'w').write(json.dumps(d,indent=2,sort_keys=True)+'\\n')\"\n"
        )
        initial_text = json.dumps(initial, indent=2, sort_keys=True) + "\n"
        expected_text = json.dumps(expected, indent=2, sort_keys=True) + "\n"
        messages = [
            _initial(instruction),
            _assistant("The file must be read before preserving its other values.", "Inspect config.json.", [(read_command, 0.1)]),
            _terminal(read_command, initial_text),
            _assistant(
                f"Only {changed} differs from the requested state.",
                "Update that JSON key without changing the others.",
                [(edit_command, 0.1)],
            ),
            _terminal(edit_command),
            _assistant("The edit completed without an error.", "Read the file once to verify the final state.", [(read_command, 0.1)], task_complete=True),
            _terminal(read_command, expected_text, confirmation=True),
            _confirm(),
        ]
        records.append(_record(task_name, "terminal-edit", instruction, messages))
    return records


def _repair_records() -> list[dict[str, Any]]:
    records = []
    for name, broken, fixed, assertions in REPAIR_TASKS:
        task_name = f"repo-{name}"
        instruction = (
            "The shell starts in the task directory and app.py is present. Inspect it, repair its bug while "
            "keeping the public function name and signature unchanged, run a focused check, and finish."
        )
        read_command = "cat app.py\n"
        edit_command = (
            "python3 -c \"p='app.py'; s=open(p).read(); "
            f"old={broken!r}; new={fixed!r}; assert s==old; open(p,'w').write(new)\"\n"
        )
        check_code = "; ".join(line for line in assertions.splitlines() if line.strip())
        check_command = f'python3 -c "{check_code}"\n'
        messages = [
            _initial(instruction),
            _assistant("The source is not visible yet, so an edit would be a guess.", "Inspect app.py first.", [(read_command, 0.1)]),
            _terminal(read_command, broken),
            _assistant(
                "The visible implementation violates the intended boundary behavior.",
                "Replace only the incorrect implementation while preserving the signature.",
                [(edit_command, 0.1)],
            ),
            _terminal(edit_command),
            _assistant(
                "The minimal source edit completed.",
                "Run focused behavioral assertions and submit if they pass.",
                [(check_command, 1.0)],
                task_complete=True,
            ),
            _terminal(check_command, confirmation=True),
            _confirm(),
        ]
        records.append(_record(task_name, "repository-repair", instruction, messages))
    return records


def build_agentic_sft_bridge(output: str | Path) -> dict[str, Any]:
    """Write a small, deterministic LlamaFactory/OpenAI-format agent-interface bridge."""

    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    records = _function_records() + _stateful_records() + _edit_records() + _repair_records()
    data_path = root / "agentic_bridge.jsonl"
    data_path.write_text("".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records))
    dataset_info = {
        "oellm_agentic_bridge": {
            "file_name": data_path.name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
    }
    info_path = root / "dataset_info.json"
    info_path.write_text(json.dumps(dataset_info, indent=2, sort_keys=True) + "\n")
    validation = validate_agentic_sft_bridge(data_path)
    manifest = {
        "schema_version": "oellm-agentic-sft-bridge-v1",
        "records": len(records),
        "categories": validation["categories"],
        "dataset_sha256": _digest(data_path),
        "dataset_info_sha256": _digest(info_path),
        "protocol": "harbor-v0.22.0/terminus-2-json",
        "protocol_template_sha256": sha256(TERMINUS_JSON_TEMPLATE.encode()).hexdigest(),
        "validation": validation,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {**manifest, "output": str(data_path), "dataset_info": str(info_path), "manifest": str(manifest_path)}


def validate_agentic_sft_bridge(path: str | Path) -> dict[str, Any]:
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    errors: list[str] = []
    categories: dict[str, int] = {}
    task_names: set[str] = set()
    assistant_turns = 0
    command_turns = 0
    for index, record in enumerate(records):
        metadata = record.get("metadata") or {}
        task_name = str(metadata.get("task_name") or "")
        category = str(metadata.get("category") or "")
        if not task_name or task_name in task_names:
            errors.append(f"record {index}: missing or duplicate task_name")
        task_names.add(task_name)
        categories[category] = categories.get(category, 0) + 1
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) < 4:
            errors.append(f"record {index}: messages must contain a multi-turn episode")
            continue
        expected_role = "user"
        for turn, message in enumerate(messages):
            role = message.get("role") if isinstance(message, dict) else None
            if role != expected_role:
                errors.append(f"record {index} turn {turn}: expected role {expected_role}, got {role}")
            expected_role = "assistant" if expected_role == "user" else "user"
            if role != "assistant":
                continue
            assistant_turns += 1
            try:
                response = json.loads(message.get("content", ""))
            except (TypeError, json.JSONDecodeError):
                errors.append(f"record {index} turn {turn}: assistant content is not one JSON object")
                continue
            if not isinstance(response.get("analysis"), str) or not isinstance(response.get("plan"), str):
                errors.append(f"record {index} turn {turn}: analysis/plan must be strings")
            commands = response.get("commands")
            if not isinstance(commands, list):
                errors.append(f"record {index} turn {turn}: commands must be a list")
                continue
            for command in commands:
                keystrokes = command.get("keystrokes") if isinstance(command, dict) else None
                if not isinstance(keystrokes, str) or not keystrokes.endswith("\n"):
                    errors.append(f"record {index} turn {turn}: command must end with newline")
                command_turns += 1
        try:
            final = json.loads(messages[-1]["content"])
        except (KeyError, TypeError, json.JSONDecodeError):
            final = {}
        if final.get("task_complete") is not True:
            errors.append(f"record {index}: final assistant turn must confirm completion")
    return {
        "ok": not errors,
        "errors": errors,
        "records": len(records),
        "assistant_turns": assistant_turns,
        "command_turns": command_turns,
        "categories": dict(sorted(categories.items())),
    }
