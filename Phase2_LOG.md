# Phase 2 Capacity Exploration Log (Qwen2-1.5B / RTX 2060 6GB / Windows)

## Environment (fixed)
- GPU: NVIDIA GeForce RTX 2060 6GB (WDDM)
- Driver: 571.96
- nvidia-smi CUDA: 12.8
- Python: 3.10
- PyTorch: 2.5.1+cu118 (CUDA runtime 11.8)
- Transformers: 4.57.3
- Accelerate: 1.12.0

## Baseline (completed)
- Model: Qwen/Qwen2-1.5B-Instruct
- Backend: Transformers + PyTorch CUDA
- Quantization: FP16
- Controls: ctx=512, max_new_tokens=64, concurrency=1
- TTFT: 0.142 s
- Throughput: 16.66 tok/s
- Peak VRAM: allocated 2.90 GB / reserved 2.94 GB
- OOM: No
- Notes: Desktop/WDDM background usage observed (~1.3GB in nvidia-smi)

---

## Experiment Template 
### EXP-20260103-2043
- Goal (single variable): Increase context length from 512 to 1024 to observe VRAM and latency impact
- Model / Backend / Quant: Qwen2-1.5B / Transformers + PyTorch CUDA / FP16
- Params: ctx=1024, max_new_tokens=64, concurrency=1
- Prompt type: single-turn (KV cache enabled, short prompt)
- Result:
  - TTFT: 0.152 s
  - Tokens/sec: 14.77 tok/s
  - Peak VRAM allocated/reserved: 2.90 GB / 2.94 GB
  - OOM: No
- Interpretation:
  Increasing ctx from 512 to 1024 did not change peak VRAM usage, indicating the actual input token length did not reach the previous ctx limit. Minor throughput drop is likely due to runtime noise or system scheduling.
- Next action:
  Design a longer prompt to explicitly reach the context boundary before further increasing ctx.

### EXP-20260103-2123
- Goal (single variable): Increase context length from 1024 to 2048 under long-input pressure (tail-preserving truncation)
- Model / Backend / Quant: Qwen2-1.5B / Transformers + PyTorch CUDA / FP16
- Params: ctx=2048, max_new_tokens=64, concurrency=1
- Prompt type: long prompt + tail-preserving truncation (keep last ctx tokens so the question is preserved)
- Result:
  - TTFT: 0.692 s
  - Tokens/sec: 11.87 tok/s
  - Peak VRAM allocated/reserved: 3.65 GB / 3.87 GB
  - OOM: No
- Interpretation:
  Increasing ctx to 2048 significantly increases prefill latency (TTFT) and VRAM usage, consistent with larger KV cache and higher attention compute cost for longer contexts. Throughput decreases accordingly.
- Next action:
  Increase ctx further (e.g., 3072) to approach the VRAM/latency boundary while keeping other variables fixed.

### EXP-20260103-2125
- Goal (single variable): Increase context length from 2048 to 3072 under long-input pressure (tail-preserving truncation)
- Model / Backend / Quant: Qwen2-1.5B / Transformers + PyTorch CUDA / FP16
- Params: ctx=3072, max_new_tokens=64, concurrency=1
- Prompt type: long prompt + tail-preserving truncation (keep last ctx tokens so the question is preserved)
- Result:
  - TTFT: 1.327 s
  - Tokens/sec: 10.31 tok/s
  - Peak VRAM allocated/reserved: 4.49 GB / 4.88 GB
  - OOM: No
- Interpretation:
  At ctx=3072, VRAM and TTFT increase sharply, indicating entry into a higher-cost region for prefill and KV cache. Reserved VRAM approaches the 6GB limit, so WDDM background usage and fragmentation may cause instability at higher ctx.
- Next action:
  Probe the boundary at ctx=4096 with all other variables fixed; if OOM occurs, treat 3072 as the FP16 safe ctx limit.

### EXP-20260103-2127
- Goal (single variable): Probe boundary at ctx=4096 under long-input pressure (tail-preserving truncation)
- Model / Backend / Quant: Qwen2-1.5B / Transformers + PyTorch CUDA / FP16
- Params: ctx=4096, max_new_tokens=64, concurrency=1
- Prompt type: long prompt + tail-preserving truncation (keep last ctx tokens so the question is preserved)
- Result:
  - TTFT: 7.781 s
  - Tokens/sec: 4.67 tok/s
  - Peak VRAM allocated/reserved: 5.65 GB / 6.37 GB
  - OOM: No (but reserved exceeded the 6GB budget)
- Interpretation:
  Although the run completed once without OOM, peak reserved VRAM exceeded the 6GB limit and TTFT/throughput degraded severely, indicating high instability risk under WDDM/background usage and allocator reservation behavior. Treat ctx=4096 as an unsafe configuration for reliable service.
- Next action:
  Set FP16 service defaults to ctx=2048 and enforce a hard max ctx=3072 for stability; keep ctx=4096 only as a boundary record.
