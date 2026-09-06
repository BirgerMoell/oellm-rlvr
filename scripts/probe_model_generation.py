#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt", default="Compute 17 + 25. End with the numeric answer.")
    args = parser.parse_args()

    model = Path(args.model).resolve(strict=True)
    tokenizer_path = Path(args.tokenizer or args.model).resolve(strict=True)
    if not os.environ.get("HF_HUB_OFFLINE") or not os.environ.get("TRANSFORMERS_OFFLINE"):
        raise RuntimeError("offline model probe requires HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")

    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=True)
    messages = [{"role": "user", "content": args.prompt}]
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_mode = "native_chat_template"
    else:
        prompt = args.prompt
        prompt_mode = "raw_prompt"
    llm = LLM(
        model=str(model),
        tokenizer=str(tokenizer_path),
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.75,
        enforce_eager=True,
        trust_remote_code=True,
    )
    output = llm.generate(
        [prompt],
        SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens),
        use_tqdm=False,
    )[0].outputs[0]
    report = {
        "ok": bool(output.text.strip()) and len(output.token_ids) > 0,
        "hostname": socket.gethostname(),
        "model": str(model),
        "tokenizer": str(tokenizer_path),
        "requested_dtype": "bfloat16",
        "torch_hip": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "prompt_mode": prompt_mode,
        "prompt": args.prompt,
        "generated_text": output.text,
        "generated_tokens": len(output.token_ids),
        "finish_reason": output.finish_reason,
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
