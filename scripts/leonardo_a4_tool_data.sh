#!/bin/bash
# Ablation A4 (qwen35-posttrain docs/OELLM_RL_RUN_PROGRAM.md): single-turn tool-call RL in thinking mode.
# Pool (verifier `tool_call`, label {"calls": [...], "mode": "thinking"}):
#   - 1,000 prompts from birgermoell/oellm-eu-tooluse-v1 @ b131d571 `grpo` (English, one gold call each);
#   - up to 500 non-English single-call prompts and 500 no-call prompts (ask_for_info, cannot_answer, no_call_direct)
#     from birgermoell/openeurollm-tool-use-v1 (the Stage-3 tool dataset), excluding its held-out bucket
#     sha256(id + "#val") < 0.005 used by the held-out tool evaluation.
# Every system turn is re-rendered with our template's tool block (tools wrapped as {"type": "function", ...}, as in
# Stage-3 training) and stored as plain system text, so TMAX renders exactly what the model was trained on.
set -euo pipefail
ROOT=${ROOT:-$SCRATCH/oellm-rl}
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
TOOLDATA=${TOOLDATA:-/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight/data/openeurollm-tool-use-v1/data}
OUT=$ROOT/data/a4-tool
mkdir -p "$OUT" "$ROOT/data/sources/oellm-eu-tooluse-v1"
singularity exec -B /leonardo_work,/leonardo_scratch "$C" env HF_HOME="$ROOT/hf-cache" python - "$ROOT" "$OUT" "$TOOLDATA" <<'PY'
import ast, collections, glob, hashlib, json, random, sys
import pyarrow as pa, pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
root, out, tooldata = sys.argv[1:]
tok = AutoTokenizer.from_pretrained(f"{root}/models/stage3-tooluse-v1-think")
rng = random.Random(20261024)

def system_text(system, tools):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": "x"}]
    text = tok.apply_chat_template(msgs, tools=tools, tokenize=False)
    assert text.startswith("<|im_start|>system\n"), text[:80]
    return text[len("<|im_start|>system\n"):text.index("<|im_end|>")]

def as_list(v):
    return v if isinstance(v, list) else ast.literal_eval(v) if isinstance(v, str) else list(v)

records = []
grpo = hf_hub_download("birgermoell/oellm-eu-tooluse-v1", "grpo/train.jsonl", repo_type="dataset",
                       revision="b131d571a9", local_dir=f"{root}/data/sources/oellm-eu-tooluse-v1")
rows = [json.loads(l) for l in open(grpo, encoding="utf-8")]
rng.shuffle(rows)
for i, r in enumerate(rows):
    msgs = as_list(r["prompt"])
    if any(m["role"] == "tool" for m in msgs) or msgs[0]["role"] != "system" or msgs[-1]["role"] != "user":
        continue
    sys_content = msgs[0]["content"]
    base, _, rest = sys_content.partition("\n\n# Tools")
    block = rest.split("<tools>\n", 1)[1].split("\n</tools>", 1)[0]
    tools = [{"type": "function", "function": json.loads(line)} for line in block.splitlines() if line.strip()]
    gold = json.loads(r["answer"]) if isinstance(r["answer"], str) else r["answer"]
    messages = [{"role": "system", "content": system_text(base, tools)}] + [{"role": m["role"], "content": m["content"]} for m in msgs[1:]]
    records.append({"id": f"grpo-{hashlib.sha256(json.dumps(msgs).encode()).hexdigest()[:16]}", "messages": messages,
                    "ground_truth": json.dumps({"calls": gold, "mode": "thinking"}), "dataset": "tool_call",
                    "kind": "grpo_single_call", "language": r["language"]})
    if sum(1 for x in records if x["kind"] == "grpo_single_call") >= 1000:
        break

heldout = lambda i: int(hashlib.sha256((i + "#val").encode()).hexdigest()[:8], 16) / 16 ** 8 < 0.005
multi, nocall = [], []
for path in sorted(glob.glob(f"{tooldata}/*.parquet")):
    for r in pq.read_table(path, columns=["id", "language", "category", "tools", "messages"]).to_pylist():
        if heldout(r["id"]):
            continue
        cat = r["category"]
        if cat in ("ask_for_info", "cannot_answer", "no_call_direct"):
            nocall.append(r)
        elif cat == "single_call" and r["language"] != "en":
            multi.append(r)

def build(r, want_call):
    msgs = as_list(r["messages"])
    tools = json.loads(r["tools"]) if isinstance(r["tools"], str) else (r["tools"] or [])
    for k, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        calls = m.get("tool_calls") or []
        if bool(calls) != want_call or msgs[k - 1]["role"] != "user":
            return None
        prompt = msgs[:k]
        if any(p["role"] == "tool" for p in prompt):
            return None
        system = prompt[0]["content"] if prompt[0]["role"] == "system" else ""
        rest = prompt[1:] if prompt[0]["role"] == "system" else prompt
        gold = []
        for c in calls:
            fn = c.get("function", c)
            args = fn.get("arguments")
            gold.append({"name": fn["name"], "arguments": json.loads(args) if isinstance(args, str) else (args or {})})
        messages = [{"role": "system", "content": system_text(system, tools)}] + [{"role": p["role"], "content": p.get("content") or ""} for p in rest]
        return {"id": r["id"], "messages": messages, "ground_truth": json.dumps({"calls": gold, "mode": "thinking"}, ensure_ascii=False),
                "dataset": "tool_call", "kind": ("multilingual_single_call" if want_call else "no_call:" + r["category"]),
                "language": r["language"]}
    return None

for pool, want, n in ((multi, True, 500), (nocall, False, 500)):
    pool.sort(key=lambda r: r["id"]); rng.shuffle(pool)
    added = 0
    for r in pool:
        rec = build(r, want)
        if rec:
            records.append(rec); added += 1
        if added >= n:
            break
rng.shuffle(records)
schema = pa.schema([("id", pa.string()), ("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
                    ("ground_truth", pa.string()), ("dataset", pa.string())])
pq.write_table(pa.Table.from_pylist([{k: r[k] for k in ("id", "messages", "ground_truth", "dataset")} for r in records], schema=schema),
               f"{out}/train.parquet")
sha = hashlib.sha256(open(f"{out}/train.parquet", "rb").read()).hexdigest()
manifest = {"kind": "a4_tool_pool", "rows": len(records), "by_kind": dict(collections.Counter(r["kind"] for r in records)),
            "languages": dict(collections.Counter(r["language"] for r in records)), "sha256": sha,
            "sources": ["birgermoell/oellm-eu-tooluse-v1@b131d571a9 grpo", "birgermoell/openeurollm-tool-use-v1 (Stage-3 data, held-out bucket excluded)"]}
json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1)
print(json.dumps(manifest, indent=1))
print("example system turn:", records[0]["messages"][0]["content"][:400])
PY
