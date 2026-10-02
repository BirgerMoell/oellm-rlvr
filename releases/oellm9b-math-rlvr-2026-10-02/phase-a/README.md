---
license: apache-2.0
base_model: birgermoell/oellm-9b-math-rlvr-canary-step16-experimental
datasets:
  - birgermoell/oellm-math-rlvr
language:
  - en
  - bg
  - cs
  - da
  - de
  - el
  - es
  - et
  - fi
  - fr
  - ga
  - hr
  - hu
  - it
  - lt
  - lv
  - mt
  - nl
  - pl
  - pt
  - ro
  - sk
  - sl
  - sv
pipeline_tag: text-generation
tags:
  - openeurollm
  - experimental
  - math
  - multilingual
  - reasoning
  - rlvr
  - dapo
---

# OpenEuroLLM 9B multilingual math RLVR, Phase A step 32 (experimental)

This is a full BF16 export after **32 additional DAPO/RLVR optimizer updates** from
[`oellm-9b-math-rlvr-canary-step16-experimental`](https://huggingface.co/birgermoell/oellm-9b-math-rlvr-canary-step16-experimental).
The earlier ancestor is the
[`Neonkraft` instruct-SFT checkpoint](https://huggingface.co/Neonkraft/oellm-9b-256k-theta64m-prelude-anneal300b-instruct-sft)
at revision `85bf18fb4f0bee6ac6270f06b1d1c6b3be200f31`. This is an **experimental
math-training example**, not an official OpenEuroLLM release or a production-ready model.
It was trained with **TMAX/Open-Instruct, not Verl**.
The `model.safetensors` file is 18,203,942,400 bytes; SHA-256:
`7f9ad2b9783263b516cdfbff338be21862c526e50e4a1a96ab9bf95d7097f4be`.

## What was trained

| Item | Exact run |
| --- | --- |
| LUMI job | `22432251`, 2026-09-30, completed `0:0` in 27m18s |
| Accelerator allocation | Two LUMI-G nodes; eight MI250X GCD learner + eight rollout, about 7.28 allocated GCD-hours |
| Algorithm | DAPO, verifiable math outcome reward, active sampling, asynchronous rollouts |
| Update shape | Eight unique prompts × eight samples, 32 updates, 2,048 accepted trajectories |
| Mixture | 80% EU-language pool, 20% English replay by sampling weight |
| Parameters | Temperature 1.0, LR `5e-7` constant, `beta=0`, BF16, DeepSpeed ZeRO-3, seed 42 |
| Backend | [`OpenEuroLLM/tmax-reproduction`](https://github.com/OpenEuroLLM/tmax-reproduction) at `3f80d37042402b8363f39c9535723b0d4cb8de54` |
| Control code | [`BirgerMoell/oellm-rlvr`](https://github.com/BirgerMoell/oellm-rlvr) at `f22ba2d18e1763474b8bd00d280c6fe135d8eaa6` |

Slurm marked one auxiliary inner step failed during setup and another cancelled during
teardown, although the top-level job completed `0:0`, all 32 learner updates finished,
and the step-32 model and restart state were written. The inner-step anomaly has not
been root-caused; this release is a training result, not a claim of flawless orchestration.

The EU component is 2,944 difficulty-1-to-3 problems: 128 per each of 23 non-English
EU official languages. The English replay is the **original 256-row canary training pool**,
not the larger 2,048-row English pool listed in the data builder's manifest. Both exact
training files are bundled as [`training_data/eu-math.parquet`](training_data/eu-math.parquet)
and [`training_data/english-pilot.parquet`](training_data/english-pilot.parquet). Their
SHA-256 values are `2c48c91df99cd1116fafd8f1f96f3ad26000711ebac5e3d3d925cfd66fc49eb6`
and `ec33cc0fcbd6be73d87ed8f0582683cc6f2e4e56297ffa7db92f3712a5ec3aff`.
The source dataset is
[`birgermoell/oellm-math-rlvr`](https://huggingface.co/datasets/birgermoell/oellm-math-rlvr)
at revision `0ffc9d6dc82717c25733b3172f4dbd63e48bab68`.

The exact rendered job config, Slurm script and selection manifest are in
[`repro/`](repro/). Their original LUMI SHA-256 values are recorded in
[`repro/SHA256SUMS`](repro/SHA256SUMS). The YAML contains allocation-specific absolute paths:
replace paths and account for a different cluster, or use the bundled Parquets and
published starting checkpoint. The original run used a LUMI AI Factory ROCm container
named `lumi-multitorch-full-u24r70f21m50t210-20260807_115122.sif`.
The EU pool excludes the earlier canary train and selection holdout by semantic group;
the full Phase A mixture deliberately replays the canary train set. The frozen
multilingual evaluation set excludes the full Phase A train set. Never
place `ground_truth` or verifier metadata in model-visible prompts.

## Paired evaluations

Both comparisons use the **same frozen prompts and deterministic generation protocol**
before and after Phase A: native chat template, temperature 0, one sample per prompt,
1,024 response-token limit, 3,072 model-token limit, seed `20260930`. The dataset,
chat-template and selected-ID hashes matched for each pair. Correctness is numeric-answer
equivalence under the project's lightweight verifier; this is not a proof-grade judge.

| Frozen set | Canary step 16 | Phase A step 32 | Paired delta, 95% bootstrap interval |
| --- | ---: | ---: | ---: |
| Multilingual procedural math, 512 prompts | 36.33% | **38.87%** | +2.54 points, [+0.20, +5.08] |
| DAPO-Math external-family diagnostic, 1,024 prompts | 4.10% | 4.59% | +0.49 points, [-0.68, +1.66] |

On the multilingual set, 27 prompts changed wrong-to-right and 14 right-to-wrong;
exact two-sided McNemar `p=0.0596`. Non-English EU prompts improved in aggregate, while
English changed by only one of 144 prompts. Think-tag use remained 100%, with zero
unbalanced tags; reasoning-channel form rose 75.20% → 77.15%, and length stops fell
7.03% → 6.45%. The DAPO diagnostic remained near its floor, with no demonstrated
improvement (`p=0.511`). Summary-only comparison artifacts are in [`eval/`](eval/);
held-out prompts and answers are not republished here.

These results support a **bounded proof of a working math RL training pipeline**, not
a general claim that this model is better at mathematics. Many prompts are procedural,
each non-English language has only 16 held-out examples, and we have not measured
safety, broad reasoning, coding, agentic behavior, or long-context retention. The
original architecture supports long context, but this RL run trained with a 3,072-token
packing limit; do not infer 256K RL performance. Read the
[`promotion plan`](https://github.com/BirgerMoell/oellm-rlvr/blob/main/docs/oellm9b-math-promotion-2026-09-28.md)
before attempting a harder phase or deployment.

## Minimal inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "birgermoell/oellm-9b-math-rlvr-phase-a-32step-experimental"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype="auto", device_map="auto")
messages = [{"role": "user", "content": "What is 37 * 48? Answer in \\boxed{}."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                       return_tensors="pt").to(model.device)
output = model.generate(inputs, max_new_tokens=512, do_sample=False)
print(tokenizer.decode(output[0, inputs.shape[-1]:], skip_special_tokens=False))
```

## License and attribution

Apache-2.0, following the parent model and the math-source dataset. The SFT ancestor
was trained on Dolci-Instruct-SFT, whose separate data terms and attribution are
described on its [model card](https://huggingface.co/Neonkraft/oellm-9b-256k-theta64m-prelude-anneal300b-instruct-sft).
This personal-account release does not imply endorsement by the OpenEuroLLM project.
The run used the LUMI supercomputer hosted by CSC in Finland.
