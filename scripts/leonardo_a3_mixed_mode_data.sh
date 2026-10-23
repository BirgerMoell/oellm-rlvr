#!/bin/bash
# Ablation A3 (qwen35-posttrain docs/OELLM_RL_RUN_PROGRAM.md): mixed-mode RL. The RL-0 think-probe pool (2,008 rows,
# thinking mode) plus 860 default-mode rows marked `rl_mode: default` on the user message (the RL mixed template then
# omits the thinking block): 430 oellm-math-rlvr train rows (12 EU languages, difficulty 1-3, native "give only the
# final answer in \boxed{}" instruction; verifier `math_direct`) and 430 exam MCQ rows from exams_qa
# (CC-BY-SA-4.0, native "answer only with the letter" instruction; verifier `mcq_letter_direct`). Rows already used in
# the probe pool or in the Stage-4 format-compliance prompts are excluded. Also builds the model directory with the
# mixed-mode template (weights hard-linked from the RL think copy).
set -euo pipefail
ROOT=${ROOT:-$SCRATCH/oellm-rl}
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
TEMPLATE=${TEMPLATE:-/leonardo_work/AIFAC_F02_352/pmoell00/sft-migration-preflight/oellm_qwen3_two_stage_v2_1_rl_mixed.jinja}
FORMAT_PROMPTS=${FORMAT_PROMPTS:-$SCRATCH/oellm-dpo/format-v1/prompts.jsonl}
MCQ=$(ls $SCRATCH/oellm-dpo/hf-cache/hub/datasets--birgermoell--oellm-eu-exam-mcq-v1/snapshots/*/grpo/train.jsonl | head -1)
OUT=$ROOT/data/a3-mixed
mkdir -p "$OUT"
src_model=$ROOT/models/stage3-tooluse-v1-think
dst_model=$ROOT/models/stage3-tooluse-v1-mixed
mkdir -p "$dst_model"
for f in "$src_model"/*; do
  case "$(basename "$f")" in chat_template.jinja) ;; *) ln -f "$f" "$dst_model/" ;; esac
done
cp "$TEMPLATE" "$dst_model/chat_template.jinja"
singularity exec -B /leonardo_work,/leonardo_scratch "$C" python - "$ROOT" "$OUT" "$FORMAT_PROMPTS" "$MCQ" "$dst_model" <<'PY'
import collections, hashlib, json, random, sys
import pyarrow as pa, pyarrow.parquet as pq
root, out, format_prompts, mcq_path, model = sys.argv[1:]
tc = json.load(open(f"{model}/tokenizer_config.json"))
assert "chat_template" not in tc, "tokenizer_config.json carries its own chat_template; it would shadow the jinja file"
rng = random.Random(20261024)
def one(value):  # oellm-math-rlvr stores ground_truth as a one-element list
    if isinstance(value, (list, tuple)):
        assert len(value) == 1, value
        value = value[0]
    return str(value)
probe = pq.read_table(f"{root}/data/think-probe/train-mixed.parquet")
used = set(probe["id"].to_pylist())
used |= {json.loads(l)["id"].split("-", 2)[-1] for l in open(format_prompts)}
rows = [dict(r, messages=[dict(m, rl_mode=None) for m in r["messages"]]) for r in probe.to_pylist()]
LANGS = ("sv", "fi", "da", "pl", "cs", "nl", "it", "pt", "ro", "el", "hu", "bg")
math = pq.read_table(f"{root}/data/sources/oellm-math-rlvr/data/train-00000-of-00001.parquet",
                     columns=["id", "messages", "ground_truth", "language", "difficulty"]).to_pylist()
pool = collections.defaultdict(list)
for r in math:
    if r["language"] in LANGS and r["difficulty"] <= 3 and r["id"] not in used:
        pool[r["language"]].append(r)
direct = []
for lang in LANGS:
    cand = sorted(pool[lang], key=lambda r: r["id"]); rng.shuffle(cand)
    for r in cand[: 430 // len(LANGS) + (1 if LANGS.index(lang) < 430 % len(LANGS) else 0)]:
        content = r["messages"][0]["content"]
        direct.append({"id": r["id"], "messages": [{"role": "user", "content": content, "rl_mode": "default"}],
                       "ground_truth": one(r["ground_truth"]), "dataset": "math_direct"})
mcq = []
for line in open(mcq_path, encoding="utf-8"):
    r = json.loads(line)
    if r["source_id"] == "exams_qa" and r["source_license"] == "CC-BY-SA-4.0" and r["id"] not in used and len(r["answer"]) == 1:
        mcq.append(r)
mcq.sort(key=lambda r: r["id"]); rng.shuffle(mcq)
for r in mcq[:430]:
    direct.append({"id": r["id"], "messages": [{"role": "user", "content": r["prompt"], "rl_mode": "default"}],
                   "ground_truth": r["answer"], "dataset": "mcq_letter_direct"})
allrows = rows + direct
rng.shuffle(allrows)
schema = pa.schema([("id", pa.string()),
                    ("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string()), ("rl_mode", pa.string())]))),
                    ("ground_truth", pa.string()), ("dataset", pa.string())])
pq.write_table(pa.Table.from_pylist(allrows, schema=schema), f"{out}/train-mixed.parquet")
sha = hashlib.sha256(open(f"{out}/train-mixed.parquet", "rb").read()).hexdigest()
manifest = {"kind": "a3_mixed_mode_pool", "rows": len(allrows), "by_dataset": dict(collections.Counter(r["dataset"] for r in allrows)),
            "default_mode_rows": sum(1 for r in allrows if r["messages"][-1]["rl_mode"] == "default"),
            "mcq_languages": dict(collections.Counter(r["id"].split("_")[3] for r in direct if r["dataset"] == "mcq_letter_direct")),
            "model_dir": model, "sha256": sha}
assert all(not r["ground_truth"].startswith("[") for r in allrows), "list-valued ground truth leaked"
json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1)
print(json.dumps(manifest, indent=1))
PY
