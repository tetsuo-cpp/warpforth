# GPT-2 Demo

Text generation with GPT-2-124M using a WarpForth-compiled attention kernel. The stock Hugging Face model is loaded normally, then `eager_attention_forward` is monkey-patched to route scaled dot-product attention through a WarpForth kernel compiled to PTX. PyCUDA shares PyTorch's CUDA context via `autoprimaryctx`, so device pointers pass directly between the two — no copies, no CPU roundtrips.

## Prerequisites

- WarpForth built locally (`cmake --build build`)
- A Vast.ai GPU instance with a PyTorch image (e.g. `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime`)

## Step 1: Compile the Kernel (Local)

```bash
./build/bin/warpforthc demo/attention.forth > demo/attention.ptx
```

A pre-compiled `attention.ptx` is included in this directory.

## Step 2: Upload to GPU Instance

```bash
scp -r demo/ demo/gpt2_generate.py root@HOST:/workspace
```

## Step 3: Install Dependencies (Remote)

```bash
pip install pycuda transformers
```

## Step 4: Generate Text (Remote)

```bash
python /workspace/gpt2_generate.py --ptx /workspace/attention.ptx --prompt "The meaning of life is"
```

| Flag | Default | Description |
|------|---------|-------------|
| `--ptx` | (required) | Path to compiled `attention.ptx` |
| `--prompt` | `"The meaning of life is"` | Input text prompt |
| `--max-tokens` | `100` | Maximum new tokens to generate |

## Limitations

- **Batch size 1** — the kernel processes one sequence at a time
- **No KV cache** — all positions are recomputed each step (`use_cache=False`)
- **Max sequence length 1024** — limited by shared memory allocation
- **12 kernel launches per layer** — one per attention head

## Files

| File | Description |
|------|-------------|
| `attention.forth` | Attention kernel source (f32 global, f64 shared) |
| `attention.ptx` | Pre-compiled PTX |
| `warpforth.py` | PyCUDA wrapper for loading and launching the kernel |
| `gpt2_generate.py` | Loads GPT-2, patches attention, generates text |
