#!/usr/bin/env python3
"""Probe Harbor's LiteLLM adapter against a local vLLM-shaped HTTP response."""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


class _Handler(BaseHTTPRequestHandler):
    request_body: dict[str, Any] | None = None

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        type(self).request_body = json.loads(self.rfile.read(length))
        body = {
            "id": "chatcmpl-oellm-probe",
            "object": "chat.completion",
            "created": 0,
            "model": "oellm-agent",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "OK"},
                    "logprobs": {
                        "content": [
                            {"token": "O", "logprob": -0.1, "bytes": [79], "top_logprobs": []},
                            {"token": "K", "logprob": -0.2, "bytes": [75], "top_logprobs": []},
                        ]
                    },
                    "finish_reason": "stop",
                    "stop_reason": None,
                    "token_ids": [101, 102],
                    "routed_experts": "probe",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            "prompt_token_ids": [11, 12, 13],
        }
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _format: str, *_args: object) -> None:
        return


async def _probe(api_base: str) -> dict[str, Any]:
    from harbor.llms.lite_llm import LiteLLM

    llm = LiteLLM(
        model_name="hosted_vllm/oellm-agent",
        api_base=api_base,
        collect_rollout_details=True,
        model_info={
            "max_input_tokens": 8192,
            "max_output_tokens": 256,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
        },
        max_tokens=2,
        timeout=30,
        max_retries=0,
    )
    response = await llm.call("probe")
    return {
        "request": _Handler.request_body,
        "response": {
            "content": response.content,
            "prompt_token_ids": response.prompt_token_ids,
            "completion_token_ids": response.completion_token_ids,
            "logprobs": response.logprobs,
            "extra": response.extra,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        report = asyncio.run(_probe(f"http://127.0.0.1:{server.server_port}/v1"))
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()

    request = report["request"] or {}
    response = report["response"]
    errors = []
    if request.get("return_token_ids") is not True:
        errors.append("LiteLLM did not forward return_token_ids=true")
    if response["prompt_token_ids"] != [11, 12, 13]:
        errors.append("Harbor did not preserve prompt_token_ids")
    if response["completion_token_ids"] != [101, 102]:
        errors.append("Harbor did not preserve completion token_ids")
    report["ok"] = not errors
    report["errors"] = errors
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
