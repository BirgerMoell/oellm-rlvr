# Experimental OpenEuroLLM 9B math-RL releases — 2026-10-02

This record separates published checkpoints and measurements from future-phase plans.
Both model cards live in this directory and on Hugging Face:

- [16-step English canary](https://huggingface.co/birgermoell/oellm-9b-math-rlvr-canary-step16-experimental),
  starting from `Neonkraft/oellm-9b-256k-theta64m-prelude-anneal300b-instruct-sft`.
- [32-step multilingual Phase A](https://huggingface.co/birgermoell/oellm-9b-math-rlvr-phase-a-32step-experimental),
  starting from the canary export.

Both used the pinned TMAX/Open-Instruct DAPO backend on LUMI, not the independent Verl
qualification. The model repositories include full BF16 weights, tokenizer and chat template,
training settings and inputs, and the aggregate paired-evaluation artifacts. Held-out prompts
are not republished. The release demonstrates completed bounded math RL training, not broad
math, long-context, safety, code, or agentic capability.
