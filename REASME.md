# Qwen2-1.5B Deployment Benchmark (Single-GPU)

A reproducible deployment and benchmarking project for **Qwen2-1.5B-Instruct** under **extreme GPU memory constraints**.

This repository focuses on **engineering correctness**, **reproducibility**, and **realistic serving behavior**, rather than model training or SOTA performance.

---

## Project Scope

- **Model**: Qwen/Qwen2-1.5B-Instruct
- **GPU**: RTX 2060 (6GB VRAM)
- **Precision**: FP16 (baseline), INT8 / INT4 (planned)
- **Backend**: PyTorch + HuggingFace Transformers
- **Serving**: FastAPI + Uvicorn
- **OS**: Windows (CUDA-enabled)

**Non-goals**
- No fine-tuning
- No multi-GPU
- No distributed inference
- No SOTA optimization claims

---

## Project Goals

1. Validate whether a 1.5B LLM can be **stably deployed** on a 6GB GPU
2. Quantify the impact of:
   - context length
   - generation length
   - concurrency
3. Identify **real serving bottlenecks** (latency, queueing, failures)
4. Provide **script-based benchmarks** with reproducible results
5. Establish a clean baseline for later quantization experiments

---

## Repository Structure
qwen2-deploy-benchmark/
├── app_min.py # FastAPI inference service (FP16)
├── bench_generate.py # Scripted concurrency benchmark
├── PHASE2_LOG.md # Context-length & KV-cache experiments
├── PHASE4_LOG.md # Serving & concurrency experiments
├── README.md # Project overview (this file)
└── requirements.txt # Minimal runtime dependencies


---

## Environment

### Python
- Python 3.10

### Core Dependencies
- torch (CUDA-enabled)
- transformers
- accelerate
- fastapi
- uvicorn
- sse-starlette (for streaming)

> The project assumes a working CUDA runtime and NVIDIA driver.

---

## Phase Overview

### Phase 1 — Minimal FP16 Inference
**Status:** Complete  
- Verified model can load and generate on RTX 2060
- Measured baseline TTFT, throughput, and VRAM usage

---

### Phase 2 — Context Length & KV Cache
**Status:** Complete  
- Systematically increased `ctx_len`
- Measured TTFT, throughput, and VRAM growth
- Identified practical context limits under 6GB VRAM

Details: `PHASE2_LOG.md`

---

### Phase 3 — Quantization (INT8 / INT4)
**Status:** Planned  
- Goal: reduce memory footprint and improve concurrency headroom
- Methods under consideration:
  - bitsandbytes INT8
  - GPTQ / AWQ (INT4)

---

### Phase 4 — Serving & Concurrency
**Status:** Complete  

- Built FastAPI-based inference service
- Implemented strict concurrency control (`Semaphore`)
- Demonstrated why **manual concurrency testing is invalid**
- Introduced script-based concurrency benchmarking
- Measured latency degradation and failure modes under load

Details: `PHASE4_LOG.md`

---

## Running the Service

```bash
uvicorn app_min:app --host 127.0.0.1 --port 8000

curl http://127.0.0.1:8000/health

python bench_generate.py \
  --concurrency 5 \
  --requests 15 \
  --ctx_len 2048 \
  --max_new_tokens 64

Key Engineering Findings (So Far)

FP16 inference is stable on RTX 2060 for Qwen2-1.5B

KV cache dominates memory growth with context length

Streaming endpoints are unsuitable for capacity benchmarking

Single-GPU concurrency primarily increases tail latency

Scripted load tests are mandatory for realistic evaluation

Reproducibility

All experiments:

Use fixed parameters

Are repeatable on the same hardware

Are documented with raw outputs and analysis

No results are inferred from single runs or manual testing.

License & Disclaimer

This project is for engineering research and benchmarking purposes only.
Model weights are subject to the original Qwen license.