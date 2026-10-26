#!/bin/bash
# T1 tool specialist pool (qwen35-posttrain docs/OELLM_RL_RUN_PROGRAM.md, merge test): the A4b multi-turn next-action
# pool (data/a4b-tool, 2,000 English rows) plus its translated slice (data/a4b-tool-translated, user turns in 16
# European languages; optional - skipped if the translation has not finished). Same schema and verifier (`tool_call`).
set -euo pipefail
ROOT=${ROOT:-$SCRATCH/oellm-rl}
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
D=$ROOT/data; OUT=$D/t1-tool; mkdir -p "$OUT"
singularity exec -B /leonardo_scratch "$C" python - "$D" "$OUT" <<'PY'
import collections, hashlib, json, os, random, sys
import pyarrow as pa, pyarrow.parquet as pq
d, out = sys.argv[1:]
base = pq.read_table(f"{d}/a4b-tool/train.parquet")
rows = base.to_pylist()
translated = []
if os.path.exists(f"{d}/a4b-tool-translated/train.parquet"):
    translated = pq.read_table(f"{d}/a4b-tool-translated/train.parquet").select(base.schema.names).cast(base.schema).to_pylist()
rows += translated
random.Random(20261026).shuffle(rows)
pq.write_table(pa.Table.from_pylist(rows, schema=base.schema), f"{out}/train.parquet")
sha = hashlib.sha256(open(f"{out}/train.parquet", "rb").read()).hexdigest()
manifest = {"kind": "t1_tool_pool", "rows": len(rows), "english": len(rows) - len(translated), "translated": len(translated), "sha256": sha}
json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1)
print(json.dumps(manifest))
PY
