# OELLM 9B math Phase B dev-g qualification — 2026-10-02

This is a **bounded qualification experiment**, not a production run or evidence of
general mathematical improvement. It starts from the released [Phase A step-32
checkpoint](https://huggingface.co/birgermoell/oellm-9b-math-rlvr-phase-a-32step-experimental)
and tests a fresh, harder, multilingual curriculum before a longer run on a production
partition. Training and evaluation status must be checked in Slurm; the profile result
alone is not a training result.

## Immutable inputs and plan

- LUMI campaign directory:
  `/scratch/project_465002530/users/bmoell/oellm-rlvr/runs/oellm9b-math-phaseb-devg-20261002`.
- Training backend: `OpenEuroLLM/tmax-reproduction` at
  `3f80d37042402b8363f39c9535723b0d4cb8de54`; training control at
  `BirgerMoell/oellm-rlvr` commit `5ee2e1e33b9c7f4d1c8a737899f1420b62b6e4cb`.
  The one-GCD profile/evaluation control is pinned at
  `f22ba2d18e1763474b8bd00d280c6fe135d8eaa6`.
- Starting export: LUMI job `22432251`, Phase A `step_32`. Training uses a fresh
  optimizer; it does not resume Phase A's optimizer/data-loader state.
- Effective selected-pool mix: 64% Phase B EU difficulty 3–5 (2,944 source rows, SHA-256
  `b8c56aca2736465154d2b07989e0cad7f41caa375f46edae043d5bf80912b74b`),
  16% Phase B English difficulty 3–5 (2,048 rows, SHA-256
  `449903ea9e8aa710a21726f6e2632c729af9af8cb994e15599d091fa2b9da9f4`),
  20% Phase A easy/medium replay (4,992 rows, SHA-256
  `e03295a0ab25b2e00a3cf79c70faf023b840fd936434f66732346fda4c1fd7d3`).
  TMAX's `weight` means a **fraction of that source file to retain**, not a final
  mixture probability. Factors `0.8696`, `0.3125`, and `0.1603` yield exactly
  2,560/640/800 selected rows, or 64/16/20 in the transformed pool. The Phase B
  builder excluded previous train/evaluation semantic groups. See the
  [phase plan](oellm9b-math-promotion-2026-09-28.md) and LUMI manifests.
- Two LUMI-G nodes, 16 MI250X GCDs (eight learner, eight rollout), `dev-g`, two-hour
  walltime. DAPO with eight prompts × eight samples, 64 optimizer updates (4,096
  accepted trajectories), LR `5e-7`, response cap 2,048, full exports and restart
  state at steps 32 and 64. The exact [YAML](../configs/lumi-grpo-math-oellm9b-phaseb-devg-64step.yaml)
  has SHA-256 `fc053389a3be4e3cae0c19cb2e64fb15fade7531b1b791d553c8c9680c730239`;
  the rendered Slurm script on LUMI has SHA-256
  `b8f64ba6dd977bb1eb5ad4cfc36699f53781e977136942bc95da179290f62506`.

## Admission profile

The [profile script](../scripts/lumi_oellm9b_math_phaseb_profile_dev_g.sbatch) ran
on one GCD as job `22487917`, completed `0:0` in 12m17s. It sampled 128 prompts
and eight attempts per prompt from each component at temperature 1 and the
2,048-token response cap, using the exact Phase A starting export.

| Component | Sample accuracy | Mixed-reward prompt groups | Length stops | Think-tag use / unbalanced |
|---|---:|---:|---:|---:|
| Harder EU | 21.48% | 40.63% | 1.07% | 100% / 0% |
| Harder English | 36.62% | 52.34% | 0.39% | 100% / 0% |
| Easy/medium replay | 29.30% | 55.47% | 1.95% | 100% / 0% |
| 64/16/20 weighted estimate | 25.47% | 45.47% | 1.14% | 100% / 0% |

The gate required every component to have 10–70% sampled accuracy, at least 20%
mixed-reward groups, at most 15% length stops, and at most 2% unbalanced tags;
the weighted mixture required at least 40% mixed groups. It passed. These
profile samples are diagnostic, not a held-out evaluation.

## Submitted jobs and interpretation

- First training attempt `22488079` was cancelled during dataset transformation,
  before any optimizer update or checkpoint, when its backend log revealed that
  nominal `0.64`/`0.16`/`0.20` factors selected 1,884/327/998 rows—roughly
  59/10/31, not the intended final mix. Dependent evaluation `22488095` was
  cancelled without running. The exact old YAML and Slurm script are preserved
  in the LUMI campaign directory as `run-cancelled-mixture.*`.
- Corrected training: job `22488151`, submitted after revalidation. The
  [rendered source configuration](../configs/lumi-grpo-math-oellm9b-phaseb-devg-64step.yaml)
  and campaign-local `run-mixfix.sbatch` are the submission record.
- Frozen paired evaluation: job `22488153`, dependent on training completing
  successfully. The [evaluation script](../scripts/lumi_oellm9b_math_phaseb_eval_dev_g.sbatch)
  requires complete steps 32 and 64, then compares each against the exact Phase A
  step-32 baseline on the disjoint 512-row multilingual holdout and 1,024-row
  DAPO-Math diagnostic. It uses the same greedy generation protocol as the
  previous Phase A evaluation.

Do not promote the step-64 export merely because its training reward improves.
The production-run decision requires finite nonzero-gradient updates, timely
checkpoint/restart, stable think-tag and length behavior, no material English or
language-family regression, and a credible paired held-out gain. The DAPO
diagnostic was near floor before this run and is primarily a regression check.
