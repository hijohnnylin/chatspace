"""Simple test script to verify streaming generation works."""

import asyncio
import sys

import torch
from chatspace.generation import VLLMSteerModel, VLLMSteeringConfig
from vllm import SamplingParams


async def test_streaming():
    """Test streaming generation."""
    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for vLLM model execution.")
        print("This test requires a GPU-enabled environment.")
        sys.exit(1)

    cfg = VLLMSteeringConfig(model_name="Qwen/Qwen3-0.6B")
    try:
        model = VLLMSteerModel(cfg)
    except Exception as e:
        print(f"ERROR: Failed to initialize model: {e}")
        print("This may be due to missing model weights or incompatible environment.")
        sys.exit(1)

    prompt = "The capital of France is"
    sampling_params = SamplingParams(temperature=0.7, max_tokens=50)

    print("Testing streaming generation:")
    print(f"Prompt: {prompt}")
    print("Streaming output:")

    async for delta in model.generate(prompt, sampling_params, stream=True):
        print(delta, end="", flush=True)

    print("\n\nTesting non-streaming generation:")
    result = await model.generate(prompt, sampling_params, stream=False)
    print(f"Full result: {result[0]}")


if __name__ == "__main__":
    asyncio.run(test_streaming())
