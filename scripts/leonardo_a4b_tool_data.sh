#!/bin/bash
# Ablation A4b (qwen35-posttrain docs/OELLM_RL_RUN_PROGRAM.md): tool-call RL on harder, multi-turn next-action items.
# A4 (single-turn, 66 % solved from the first step) did not move the held-out tool check; this pool asks for the next
# action inside real multi-step tool conversations (earlier calls and results in the prompt). Verifier `tool_call`,
# label {"calls": [...], "mode": "thinking"} (an expected plain reply = no-call gold: reward 1 only without a call).
#   - nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1 @ bfc7a4dd (CC-BY-4.0): policy-following service agent,
#     ~17 tools; rows kept only with Qwen3-235B pass rate in [0.1, 0.8] (informative groups);
#   - nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1 @ 0ac34ec8 (CC-BY-4.0): multi-step API trajectories;
#   - MadeAgents/xlam-irrelevance-7.5k @ 34323bf0 (CC-BY-4.0): no tool fits -> answer without calling.
# Conversion: Responses-API items -> chat messages; earlier `reasoning` items are dropped; consecutive function_call
# items become one assistant turn rendered exactly as our template renders tool_calls; function_call_output -> role
# tool. Tools are rendered into the system turn with our template (as in A4) and stored as plain text. Prompts longer
# than MAX_PROMPT tokens (rendered with the model's template) are dropped. English only (as the sources).
# Downloads: run once on a login node (internet); the job itself is offline.
set -euo pipefail
ROOT=${ROOT:-$SCRATCH/oellm-rl}
MODEL=${MODEL:?model dir whose chat template renders the pool (e.g. $ROOT/models/stage4-dpo-v2-mixed)}
MAX_PROMPT=${MAX_PROMPT:-4096}
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
SRC=$ROOT/data/sources/tool-rl
OUT=$ROOT/data/a4b-tool
mkdir -p "$OUT"
singularity exec -B /leonardo_work,/leonardo_scratch "$C" python - "$SRC" "$OUT" "$MODEL" "$MAX_PROMPT" <<'PY'
import collections, hashlib, json, random, sys
import pyarrow as pa, pyarrow.parquet as pq
from transformers import AutoTokenizer
src, out, model, max_prompt = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
tok = AutoTokenizer.from_pretrained(model)
rng = random.Random(20261025)
WANT = {"conversational": 1200, "function_calling": 500, "irrelevance": 300}
REVISIONS = {"conversational": "nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1@bfc7a4dd55ec1424cc3761ceacdd85a5743b4f4e",
             "function_calling": "nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1@0ac34ec81bb6976d3746407df306a57ce48f0bd8",
             "irrelevance": "MadeAgents/xlam-irrelevance-7.5k@34323bf09efc7e4a394998a0fa91ff997617c369"}
skipped = collections.Counter()

def system_text(system, tools):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": "x"}]
    text = tok.apply_chat_template(msgs, tools=tools, tokenize=False)
    assert text.startswith("<|im_start|>system\n"), text[:80]
    return text[len("<|im_start|>system\n"):text.index("<|im_end|>")]

def call_text(name, arguments):
    args = json.dumps(json.loads(arguments) if isinstance(arguments, str) else (arguments or {}), ensure_ascii=False)
    return '<tool_call>\n{"name": "' + name + '", "arguments": ' + args + '}\n</tool_call>'

def text_of(content):
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content or [] if isinstance(part, dict))

def convert(row, kind):
    rcp = row["responses_create_params"]
    tools = [{"type": "function", "function": {k: t[k] for k in ("name", "description", "parameters") if k in t}}
             for t in rcp.get("tools") or []]
    system, msgs = "", []
    for item in rcp["input"]:
        kind_ = item.get("type")
        role = item.get("role")
        if kind_ == "reasoning":
            continue
        if role == "system":
            system = text_of(item.get("content"))
        elif role == "user":
            msgs.append({"role": "user", "content": text_of(item.get("content"))})
        elif role == "assistant":
            msgs.append({"role": "assistant", "content": text_of(item.get("content")).strip()})
        elif kind_ == "function_call":
            text = call_text(item["name"], item.get("arguments"))
            if msgs and msgs[-1]["role"] == "assistant":
                msgs[-1]["content"] = (msgs[-1]["content"] + "\n" if msgs[-1]["content"] else "") + text
            else:
                msgs.append({"role": "assistant", "content": text})
        elif kind_ == "function_call_output":
            msgs.append({"role": "tool", "content": text_of(item.get("output"))})
        else:
            return None, "unknown_item"
    if not msgs or msgs[-1]["role"] not in ("user", "tool"):
        return None, "bad_last_turn"
    if any("<think>" in m["content"] or "</think>" in m["content"] for m in msgs):
        return None, "think_tag_in_history"
    action = row["expected_action"]
    if action["type"] == "function_call":
        gold = [{"name": action["name"], "arguments": json.loads(action.get("arguments") or "{}")}]
    elif action["type"] == "message":
        gold = []
    else:
        return None, "unknown_action"
    return ([{"role": "system", "content": system_text(system, tools)}] + msgs, gold), None

def irrelevance_rows(path):
    for r in json.load(open(path, encoding="utf-8")):
        tools = [{"type": "function", "function": {"name": t["name"], "description": t.get("description", ""),
                  "parameters": {"type": "object", "properties": t.get("parameters", {})}}} for t in json.loads(r["tools"])]
        if json.loads(r["answers"]) != []:
            continue
        yield ([{"role": "system", "content": system_text("", tools)}, {"role": "user", "content": r["query"]}], []), r["query"]

def keep(messages):
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    ids = ids["input_ids"] if isinstance(ids, dict) else ids
    return len(ids) <= max_prompt, len(ids)

records = []
lengths = collections.defaultdict(list)
for kind, fname in (("conversational", "Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1/train.jsonl"),
                    ("function_calling", "Nemotron-RL-Agentic-Function-Calling-Pivot-v1/train.jsonl")):
    rows = []
    with open(f"{src}/{fname}", encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            if kind == "conversational" and not (0.1 <= r.get("pass_rate", -1) <= 0.8):
                skipped[f"{kind}:pass_rate"] += 1
                continue
            rows.append(r)
    rng.shuffle(rows)
    added = 0
    for r in rows:
        converted, why = convert(r, kind)
        if converted is None:
            skipped[f"{kind}:{why}"] += 1
            continue
        messages, gold = converted
        ok, n = keep(messages)
        if not ok:
            skipped[f"{kind}:too_long"] += 1
            continue
        key = json.dumps(messages, ensure_ascii=False)
        records.append({"id": f"{kind}-{hashlib.sha256(key.encode()).hexdigest()[:16]}", "messages": messages,
                        "ground_truth": json.dumps({"calls": gold, "mode": "thinking"}, ensure_ascii=False),
                        "dataset": "tool_call", "kind": kind + (":call" if gold else ":reply")})
        lengths[kind].append(n)
        added += 1
        if added >= WANT[kind]:
            break
irr = list(irrelevance_rows(f"{src}/xlam-irrelevance-7.5k/xlam-7.5k-irrelevancek.json"))
rng.shuffle(irr)
for (messages, gold), query in irr[: WANT["irrelevance"]]:
    ok, n = keep(messages)
    if not ok:
        continue
    records.append({"id": "irrelevance-" + hashlib.sha256(query.encode()).hexdigest()[:16], "messages": messages,
                    "ground_truth": json.dumps({"calls": [], "mode": "thinking"}), "dataset": "tool_call",
                    "kind": "irrelevance:reply"})
    lengths["irrelevance"].append(n)
rng.shuffle(records)
schema = pa.schema([("id", pa.string()), ("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
                    ("ground_truth", pa.string()), ("dataset", pa.string())])
pq.write_table(pa.Table.from_pylist([{k: r[k] for k in ("id", "messages", "ground_truth", "dataset")} for r in records], schema=schema),
               f"{out}/train.parquet")
sha = hashlib.sha256(open(f"{out}/train.parquet", "rb").read()).hexdigest()
pct = lambda v, q: sorted(v)[int(q * (len(v) - 1))] if v else None
manifest = {"kind": "a4b_tool_pool", "rows": len(records), "by_kind": dict(collections.Counter(r["kind"] for r in records)),
            "prompt_tokens": {k: {"p50": pct(v, 0.5), "p95": pct(v, 0.95), "max": max(v)} for k, v in lengths.items()},
            "max_prompt_tokens": max_prompt, "skipped": dict(skipped), "sources": REVISIONS, "template_model": model,
            "sha256": sha}
json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1)
print(json.dumps(manifest, indent=1))
ex = next(r for r in records if r["kind"].startswith("function_calling"))
print("example (tail):", json.dumps(ex["messages"][-3:], ensure_ascii=False)[:900], "\ngold:", ex["ground_truth"][:300])
PY
