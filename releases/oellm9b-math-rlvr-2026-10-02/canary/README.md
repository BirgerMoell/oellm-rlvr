---
license: apache-2.0
base_model: Neonkraft/oellm-9b-256k-theta64m-prelude-anneal300b-instruct-sft
datasets:
  - birgermoell/oellm-math-rlvr
language:
  - en
pipeline_tag: text-generation
tags:
  - openeurollm
  - experimental
  - math
  - reasoning
  - rlvr
  - dapo
---

# OpenEuroLLM 9B math-RL canary, step 16 (experimental)

This is the **16-update starting checkpoint** for the separate
[`oellm-9b-math-rlvr-phase-a-32step-experimental`](https://huggingface.co/birgermoell/oellm-9b-math-rlvr-phase-a-32step-experimental)
release. It is a full BF16 model export, not an adapter. It starts from
[`Neonkraft/oellm-9b-256k-theta64m-prelude-anneal300b-instruct-sft`](https://huggingface.co/Neonkraft/oellm-9b-256k-theta64m-prelude-anneal300b-instruct-sft)
at revision `85bf18fb4f0bee6ac6270f06b1d1c6b3be200f31`.

This is research evidence that a small, verifier-backed RL run can complete on LUMI. It is **not**
an official OpenEuroLLM release, a general math benchmark winner, or evidence of safety or
256K-context quality. It was trained with **TMAX/Open-Instruct DAPO, not Verl**.
The `model.safetensors` file is 18,203,942,400 bytes; SHA-256:
`c5b01a4b13d5bd76fdb4a69e73994e73704bb73e62a6a067068032f46a3f7565`.

## Training

- Source: [`birgermoell/oellm-math-rlvr`](https://huggingface.co/datasets/birgermoell/oellm-math-rlvr)
  at revision `0ffc9d6dc82717c25733b3172f4dbd63e48bab68`.
- Training pool: 256 English difficulty-1-to-3 problems. The exact selected Parquet is
  [`training_data/english-pilot.parquet`](training_data/english-pilot.parquet), SHA-256
  `ec33cc0fcbd6be73d87ed8f0582683cc6f2e4e56297ffa7db92f3712a5ec3aff`.
- Algorithm: DAPO with outcome-verifiable math reward; 16 optimizer updates, eight unique
  prompts by eight stochastic samples per update, temperature 1.0, active sampling.
- Optimizer: constant learning rate `5e-7`, `beta=0`, BF16, DeepSpeed ZeRO-3,
  gradient checkpointing; seed 42.
- Topology: two LUMI-G nodes, eight MI250X GCDs for the learner and eight for rollout
  engines. Job `22398668` completed in 14m30s, about 3.87 allocated GCD-hours.
- Exact control/config: [`BirgerMoell/oellm-rlvr`](https://github.com/BirgerMoell/oellm-rlvr)
  and [`repro/canary-run.yaml`](repro/canary-run.yaml) (SHA-256
  `c55f686197dbbee6961a3f2f734b488a9379f566e21c54fb7116c001864f0974`).
  Backend: [`OpenEuroLLM/tmax-reproduction`](https://github.com/OpenEuroLLM/tmax-reproduction)
  at `3f80d37042402b8363f39c9535723b0d4cb8de54`.

The YAML records LUMI-specific absolute paths and container image. To rerun, replace the
project account, checkpoint and data paths for your allocation while preserving the recorded
data hashes and training parameters. `ground_truth` and verifier metadata must never be
included in model-visible prompts.

## Evidence and limits

All 16 updates had finite, nonzero gradients; 128/128 accepted prompt groups had mixed
rewards, and a cold restart from saved learner state completed an additional update. On a
frozen 1,024-prompt English procedural holdout, greedy exact-answer accuracy was 42.97%
for the SFT parent and 44.63% for this canary (+1.66 percentage points; paired bootstrap
95% interval +0.20 to +3.13 points). This holdout is drawn from the same procedural
generator collection; step 16 was selected after inspecting step-8 and step-16
results, so the interval is exploratory rather than an untouched confirmation.
This is not an independent general-math estimate. See the
[`canary qualification record`](https://github.com/BirgerMoell/oellm-rlvr/blob/main/docs/qualification-oellm9b-instruct-math-canary-2026-09-28.md).

The model can still be wrong, overfit procedural templates, or produce unbalanced or
unsound reasoning. Think-tag structure is not proof of correct reasoning. No safety,
code, agentic, or long-context retention evaluation is claimed here.

## License and attribution

Apache-2.0, following the parent model and the math-source dataset. The SFT parent was
trained on Dolci-Instruct-SFT, whose separate data terms and attribution are described on
its [model card](https://huggingface.co/Neonkraft/oellm-9b-256k-theta64m-prelude-anneal300b-instruct-sft).
OpenEuroLLM and LUMI/CSC contributors are acknowledged; this personal-account release
does not imply project endorsement.
