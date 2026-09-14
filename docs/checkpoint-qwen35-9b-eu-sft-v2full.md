# Qwen3.5-9B `EU-SFT-v2full` checkpoint record

This is the Qwen control checkpoint used by the OpenEuroLLM RLVR qualification campaign. It is a
full-parameter supervised fine-tune of the text-only `Qwen3.5-9B` base. It is not a LoRA adapter,
not a DPO checkpoint, and not an OpenEuroLLM model checkpoint.

## Identity and artifacts

- Base: the LUMI snapshot at
  `/scratch/project_465002530/users/bmoell/models/Qwen3.5-9B`.
- Training output:
  `/scratch/project_465002530/users/bmoell/qwen35-posttrain/output/qwen35-9b-eu-sft-v2full`.
- RL/inference export:
  `/scratch/project_465002530/users/bmoell/qwen35-posttrain/output/qwen35-9b-eu-sft-v2full-bf16-textfix`.
- Architecture: `Qwen3_5ForCausalLM` / `qwen3_5_text`, 32 layers, hidden size 4,096,
  intermediate size 12,288, 16 attention heads, 4 key/value heads, and a 248,320-token vocabulary.
- The architecture advertises a 262,144-token maximum position range. This SFT run used 4,096-token
  packed sequences, so the run is not evidence of retained 256K instruction-following quality.

The original training directory is about 134 GB because it contains the final model and three
training-state checkpoints. Its serialized model tensors are FP32. The derived inference export is
an audited BF16 cast with 427/427 tensors present, five safetensor shards, and about 17 GB on disk.
It excludes optimizer, scheduler, RNG, and nested checkpoint artifacts. The derived export is the
immutable starting point used by the RL control runs.

## SFT data

The effective source pool contained 1,332,196 conversations:

| Slice | Rows | Share | Purpose |
|---|---:|---:|---|
| Tulu-3 commercial staging subset | 920,552 | 69.10% | Predominantly English general-instruction replay |
| EuroBlocks staged subset | 161,644 | 12.13% | Synthetic European-language instruction tuning |
| Prefix of Nemotron Post-Training Dataset v2 math | 250,000 | 18.77% | English `<think>` math reasoning |

The first two slices form the 1,082,196-row general Parquet. The staging name `85-15` refers to its
85.06% Tulu / 14.94% EuroBlocks row ratio; it does not mean that 85% of the full run is European
language data. The reasoning Parquet concatenated a 1,426,602-row Nemotron-v2 math source before a
100,000-row Llama-Nemotron sample described as including Finnish. The YAML selected
`train[:250000]`, so the effective reasoning slice came entirely from the first, English Nemotron
source. This checkpoint therefore has multilingual general SFT but not pan-European reasoning SFT.

## Training recipe and outcome

- Framework: TRL `0.28.0`, Transformers `5.12.1`, PyTorch `2.9.1+rocm6.4`, Datasets `5.0.0`,
  Tokenizers `0.22.2`.
- Method: full-parameter `SFTTrainer`, BF16 compute, gradient checkpointing, best-fit packing at
  4,096 tokens.
- Schedule: 3,000 optimizer steps, per-rank batch 1, gradient accumulation 4, global batch 128,
  cosine learning-rate schedule, 3% warmup, peak learning rate `6e-6`, and no weight decay.
- Final state: epoch `0.64956`, 1,564,888,076 sampled tokens, loss `0.5600`, mean token accuracy
  `0.83038`, and gradient norm `0.34467`.
- Slurm job `19549599` completed on 2 July 2026 in 17:14:47. It launched 32 workers over eight
  LUMI-G nodes (four workers per node). The allocation reserved all 64 MI250 GCDs, however, so it
  consumed about 1,104 allocated GCD-hours while only about 552 worker-GCD-hours performed model
  training. Future runs should request the intended four GCDs per node or use all eight.

There was no held-out validation split in the SFT run. The low final loss is evidence of stable
optimization, not evidence that multilingual reasoning improved. Downstream capability, language
consistency, and retention must be established by frozen evaluations.

## What the RL qualification has established

The native Qwen chat template supplies the opening `<think>` token sequence in the generation
prompt. The model usually emits the reasoning suffix, `</think>`, and a boxed answer. Evaluators
must reconstruct the template-provided opener before judging tag balance.

At a 1,536-token response cap on a 36-language basic symbolic profile, the checkpoint reached
90.28% sample accuracy and 100% pass@8, with 9.38% length stops. That pool was too easy for useful
RL: 69.44% of prompt groups had zero reward variance. A deliberately harder profile produced ample
variance but exposed family-specific verbosity. The current campaign therefore profiles and admits
task families before allocating learner GPUs; the checkpoint name alone is not treated as proof that
an RL curriculum is appropriate.

The checkpoint is valuable as a same-scale control for testing the LUMI rollout, verifier, optimizer,
weight-transfer, and multilingual scheduling machinery. A successful control run does not replace
the required profile and evaluation of the eventual OpenEuroLLM SFT/DPO predecessor.
