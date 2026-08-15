# GPT-2 demo

The GPT-2 demo performs text generation with GPT-2-124M using a WarpForth-compiled attention kernel.
It loads the stock Hugging Face model, then patches `eager_attention_forward` to route scaled dot-product attention through a WarpForth kernel compiled to PTX.
PyCUDA shares PyTorch's CUDA context through `autoprimaryctx`, so device pointers pass directly between the two without copies or CPU round trips.

## Prerequisites

- WarpForth built locally with `cmake --build build`
- A Vast.ai GPU instance with a PyTorch image such as `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime`

## Compile the kernel locally

```bash
./build/bin/warpforthc demo/attention.forth > demo/attention.ptx
```

A precompiled `attention.ptx` is included in the `demo` directory.

## Upload the demo

```bash
scp -r demo/ root@HOST:/workspace
```

## Install remote dependencies

```bash
pip install pycuda transformers
```

## Generate text

```bash
python /workspace/demo/gpt2_generate.py \
  --ptx /workspace/demo/attention.ptx \
  --prompt "The meaning of life is"
```

| Flag | Default | Description |
|---|---|---|
| `--ptx` | required | Path to the compiled `attention.ptx`. |
| `--prompt` | `"The meaning of life is"` | Input text prompt. |
| `--max-tokens` | `100` | Maximum number of new tokens. |

## Limitations

- **Batch size 1:** the kernel processes one sequence at a time.
- **No KV cache:** all positions are recomputed at each step with `use_cache=False`.
- **Maximum sequence length 1024:** shared-memory allocation limits the sequence length.
- **12 kernel launches per layer:** one launch is made for each attention head.

## Files

| File | Description |
|---|---|
| `attention.forth` | Attention kernel source using f32 global and f64 shared memory. |
| `attention.ptx` | Precompiled PTX. |
| `warpforth.py` | PyCUDA wrapper for loading and launching the kernel. |
| `gpt2_generate.py` | Loads GPT-2, patches attention, and generates text. |
