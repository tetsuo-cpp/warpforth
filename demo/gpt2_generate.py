"""GPT-2 text generation with WarpForth-compiled attention kernel.

Loads GPT-2-124M from Hugging Face, monkey-patches the attention mechanism
to use a WarpForth CUDA kernel, and generates text from a prompt.

Usage:
    # Local: compile kernel to PTX
    ./build/bin/warpforthc demo/attention.forth > demo/attention.ptx

    # Remote (Vast.ai GPU instance):
    pip install pycuda transformers torch
    python gpt2_generate.py --ptx attention.ptx --prompt "The meaning of life is"
"""

from __future__ import annotations

import argparse

import torch
import transformers.models.gpt2.modeling_gpt2 as gpt2_module
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from warpforth import AttentionKernel


def make_warpforth_eager_attn(attn_kernel: AttentionKernel):
    """Create a replacement for eager_attention_forward using the WarpForth kernel.

    The transformers eager_attention_forward signature is:
        (module, query, key, value, attention_mask, **kwargs)
    query/key/value: (batch, n_heads, seq_len, head_dim) float32 CUDA.
    Returns (attn_output, attn_weights) with attn_output transposed to
    (batch, seq_len, n_heads, head_dim).
    """

    def warpforth_eager_attn(module, query, key, value, attention_mask=None, **kwargs):
        _batch, n_heads, seq_len, head_dim = query.shape
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        attn_output = torch.zeros_like(query)

        for h in range(n_heads):
            attn_kernel(
                query[0, h],
                key[0, h],
                value[0, h],
                attn_output[0, h],
                seq_len,
                head_dim,
            )

        return attn_output.transpose(1, 2), None

    return warpforth_eager_attn


def main():
    parser = argparse.ArgumentParser(description="GPT-2 generation with WarpForth attention")
    parser.add_argument("--ptx", required=True, help="Path to compiled attention.ptx")
    parser.add_argument("--prompt", default="The meaning of life is", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max new tokens to generate")
    args = parser.parse_args()

    print("Loading GPT-2 model...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager").cuda().float()
    model.eval()

    print(f"Loading WarpForth attention kernel from {args.ptx}")
    attn_kernel = AttentionKernel(args.ptx)

    inputs = tokenizer(args.prompt, return_tensors="pt").to("cuda")
    print(f"Prompt: {args.prompt!r} ({inputs['input_ids'].shape[1]} tokens)")

    gpt2_module.eager_attention_forward = make_warpforth_eager_attn(attn_kernel)

    print(f"\nGenerating (max {args.max_tokens} new tokens)...")
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_tokens,
            use_cache=False,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\n{'=' * 60}")
    print(output_text)
    print(f"{'=' * 60}")
    print(f"({output_ids.shape[1]} tokens total)")


if __name__ == "__main__":
    main()
