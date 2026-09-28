# OELLM 9B multilingual math promotion — 2026-09-28

## Objective and checkpoint flow

Promote the selected 16-update English math canary through two independently gated 32-update windows:

```text
canary step 16
  -> phase A: EU24 difficulty 1–3 + 20% proven English replay
  -> frozen evaluation and fresh eight-sample profile
  -> phase B: novel EU24/English difficulty 3–5 + 20% phase-A replay
  -> frozen evaluation and checkpoint selection
```

Each phase starts a fresh optimizer from the selected exported model at the preceding boundary. This is
intentional: changing a TMAX dataset mixer while restoring its DeepSpeed/data-loader state would make the
curriculum boundary ambiguous. The full learner state remains available for restart *within* each phase.

## Frozen starting point

- input checkpoint: step 16 from LUMI job `22398668`;
- parent model: `Neonkraft/oellm-9b-256k-theta64m-prelude-anneal300b-instruct-sft` at
  `85bf18fb4f0bee6ac6270f06b1d1c6b3be200f31`;
- math source: `birgermoell/oellm-math-rlvr` at
  `0ffc9d6dc82717c25733b3172f4dbd63e48bab68`;
- backend: `OpenEuroLLM/tmax-reproduction` at
  `3f80d37042402b8363f39c9535723b0d4cb8de54`;
- topology: two LUMI-G nodes, eight learner GCDs and eight TP=1 rollout GCDs;
- update shape: eight prompts by eight stochastic samples, DAPO, constant `5e-7` learning rate;
- selection rule: training reward never selects a checkpoint.

## Frozen datasets

The data builder records every exclusion by path, SHA-256, and semantic-group count. Phase A excludes the
original 256-row canary train set and its 1,024-row selection holdout. Phase B additionally excludes every
phase-A group. The new 512-row multilingual holdout excludes both training phases and both previous frozen
pools. Parallel translations of a semantic problem cannot cross these boundaries.

| Artifact | Rows | Role |
|---|---:|---|
| Phase A EU pool, difficulty 1–3 | 2,944, 128 per non-English EU official language | 80% warm/core training |
| Original English canary pool | 256 | 20% known-learnable replay in phase A |
| Phase B EU pool, difficulty 3–5 | 2,944 | harder multilingual training |
| Phase B English pool, difficulty 3–5 | 2,048 | harder English training |
| Phase A combined pool | 4,992 | 20% easy/medium replay in phase B |
| Frozen multilingual holdout, difficulty 1–5 | 512, including 16 per non-English EU language | paired selection only |
| Frozen DAPO-Math evaluation | 1,024 | non-procedural external-family diagnostic only |

The generated manifests are authoritative for the exact hashes. The procedural holdout is useful for paired
measurement but does not replace the DAPO external-family diagnostic. DAPO was near the starting checkpoint's
floor, so it is a regression/large-improvement signal rather than the sole selector.

## Gates and schedule

Before phase A, profile 256 phase-A EU prompts and all 256 replay prompts with eight samples each. Require:

- 10–70% sample accuracy in the weighted mixture;
- at least 20% mixed-reward groups in each source and at least 40% in the weighted mixture;
- at most 15% length stops and 2% inference/verifier errors;
- think-tag use remains 100%, with no increase in unbalanced tags.

Run phase A for 32 updates only after those gates pass. Save model and full state at updates 16 and 32. Require
finite, non-zero gradients on at least 95% of updates, maximum policy lag four, and no source or language
disappearing from the realized rollouts.

At the boundary, evaluate phase-A steps 16 and 32 on exactly the frozen 512-row multilingual holdout and the
1,024-row DAPO holdout. Re-profile the phase-B EU, English, and replay components with eight samples per prompt.
Only then materialize the phase-B config from the selected phase-A export. Use 64% novel harder EU, 16% novel
harder English, and 20% phase-A replay for another 32 updates.

Final promotion requires a positive paired multilingual-holdout delta with a bootstrap interval that does not
show a material regression, no meaningful English or language-family regression, stable response form and
length, and no regression on the DAPO external-family diagnostic. A checkpoint that fails is rejected even if
its online reward improved.

## Resource estimate

At the measured canary throughput, each 32-update two-node phase should use roughly 8–12 MI250 GCD-hours.
Profiles and frozen one-GCD evaluations add about 2–4 GCD-hours. The complete gated 64-update study is expected
to remain around 20–30 GCD-hours; stop after phase A if its admission, training, or evaluation gates fail.
