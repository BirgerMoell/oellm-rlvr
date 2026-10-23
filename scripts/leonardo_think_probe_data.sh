#!/bin/bash
# Build the Leonardo thinking-mode RL probe pool (login node; network needed for the HF downloads).
# Exact counts in one parquet: 1,000 EU-language (12 x ~84) + 400 English oellm-math-rlvr train rows at
# difficulty 1-3, and 600 GSM8K train rows (natural prompt; the template already opens <think>).
set -euo pipefail
ROOT=${ROOT:-$SCRATCH/oellm-rl}
C=/leonardo_work/OELLM_prod2026/container_images/post-training-25.10-transformers-olmo3-patch.sif
PYLIBS=${OELLM_WORK:?source leonardo.env}/europa/pylibs
OUT=$ROOT/data/think-probe; SRC=$ROOT/data/sources
mkdir -p "$OUT" "$SRC"
py() { singularity exec -B /leonardo_work,/leonardo_scratch "$C" env PYTHONPATH="$ROOT/oellm-rlvr/src:$PYLIBS" HF_HOME="$ROOT/hf-cache" python "$@"; }
py - "$SRC" <<'PY'
import sys
from huggingface_hub import hf_hub_download
src = sys.argv[1]
print(hf_hub_download("birgermoell/oellm-math-rlvr", "data/train-00000-of-00001.parquet", repo_type="dataset",
                      revision="0ffc9d6dc82717c25733b3172f4dbd63e48bab68", local_dir=f"{src}/oellm-math-rlvr"))
for split in ("train", "test"):
    print(hf_hub_download("openai/gsm8k", f"main/{split}-00000-of-00001.parquet", repo_type="dataset",
                          revision="740312add88f781978c0658806c59bc2815b9866", local_dir=f"{src}/gsm8k"))
PY
MATH=$SRC/oellm-math-rlvr/data/train-00000-of-00001.parquet
for lang in sv fi da pl cs nl it pt ro el hu bg; do
  py -m oellm_rlvr.cli sample-math --source "$MATH" --output "$OUT/parts/eu-$lang.parquet" --count 84 --language $lang \
     --min-difficulty 1 --max-difficulty 3 --seed 20261022
done
py -m oellm_rlvr.cli sample-math --source "$MATH" --output "$OUT/parts/en.parquet" --count 400 --language en \
   --min-difficulty 1 --max-difficulty 3 --seed 20261022
py -m oellm_rlvr.cli prepare-gsm8k --train-source "$SRC/gsm8k/main/train-00000-of-00001.parquet" \
   --test-source "$SRC/gsm8k/main/test-00000-of-00001.parquet" --output-dir "$OUT/gsm8k" \
   --revision 740312add88f781978c0658806c59bc2815b9866 --prompt-style natural
py - "$OUT" <<'PY'
import sys, glob, random
import pyarrow as pa, pyarrow.parquet as pq
out = sys.argv[1]
cols = ["id", "messages", "ground_truth", "dataset"]
parts = [pq.read_table(p).select(cols) for p in sorted(glob.glob(f"{out}/parts/*.parquet"))]
eu_en = pa.concat_tables(parts, promote_options="permissive")
gsm = pq.read_table(glob.glob(f"{out}/gsm8k/*train*.parquet")[0]).select(cols)
rng = random.Random(20261022)
gsm = gsm.take(sorted(rng.sample(range(gsm.num_rows), 600)))
mixed = pa.concat_tables([eu_en, gsm.cast(eu_en.schema)])
mixed = mixed.take(rng.sample(range(mixed.num_rows), mixed.num_rows))
pq.write_table(mixed, f"{out}/train-mixed.parquet")
langs = {}
for i in mixed["id"].to_pylist():
    k = "gsm8k" if "gsm8k" in i else "oellm"
    langs[k] = langs.get(k, 0) + 1
print("train-mixed rows:", mixed.num_rows, langs, "| dataset keys:", set(mixed["dataset"].to_pylist()))
print("example:", mixed.slice(0, 1).to_pylist()[0])
PY
