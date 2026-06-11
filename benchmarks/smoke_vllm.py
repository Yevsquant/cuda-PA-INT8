"""Day-0 smoke test: confirm vLLM runs a single generation on the A100.

Run:
    source env.sh
    python benchmarks/smoke_vllm.py
"""

from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

PROMPTS = [
    "The capital of France is",
    "Write one sentence explaining what PagedAttention does:",
]


def main() -> None:
    llm = LLM(
        model=MODEL,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        max_model_len=4096,
    )
    params = SamplingParams(temperature=0.0, max_tokens=64)
    outputs = llm.generate(PROMPTS, params)
    for out in outputs:
        print("=" * 60)
        print("PROMPT :", out.prompt)
        print("OUTPUT :", out.outputs[0].text.strip())
    print("=" * 60)
    print("smoke test OK")


if __name__ == "__main__":
    main()
