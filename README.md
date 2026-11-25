# Lightweight RAG

## Setup

```bash
python -m venv rag_env
source rag_env/bin/activate
pip install --upgrade pip
pip install "transformers>=4.46" "accelerate>=0.26.0" "sentence-transformers" "scikit-learn" "torch" "bitsandbytes"  # bitsandbytes optional (CUDA)
```

If using gated models, set `HUGGING_FACE_HUB_TOKEN=...`.

## Model choice
Set `MODEL_NAME` in `lightweight_rag.py`, e.g.:
- `mistralai/Mistral-7B-Instruct-v0.2` (better quality; heavier)
- `microsoft/phi-2` (lighter)
- `meta-llama/Llama-3.2-3B-Instruct` (gated; requires HF access token)

## Device selection
In `lightweight_rag.py`, choose:
- `DEVICE_MODE = "auto"` (default: MPS on Apple, CUDA on Nvidia, CPU fallback)
- `DEVICE_MODE = "mps"` to force Apple GPU
- `DEVICE_MODE = "cuda"` to force Nvidia GPU (uses 4-bit via bitsandbytes)

For MPS fallback on missing ops:
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Run
- Provide query via flag (queries the corpus for RAG):
  ```bash
  python lightweight_rag.py -q "Why cache model weights locally?"
  ```
- Or run and type when prompted (queries the corpus for RAG):
  ```bash
  python lightweight_rag.py
  Enter your question (will query the corpus for RAG): Why cache model weights locally?
  ```

## Notes
- 7B in fp16 on MPS fits in 24 GB; for tighter memory on CUDA, 4-bit is enabled automatically when `DEVICE_MODE="cuda"`.
- If you see “gated repo” errors, request access on the model page or switch to an ungated model.
- The CLI shows a spinner during generation and prints inference time.
```// filepath: /Users/semsion/Documents/Projects/Python/RAG/20251209/python-rag-pipeline/README.md



