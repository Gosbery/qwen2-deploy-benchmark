import time
import asyncio
import json
import threading
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)
from sse_starlette.sse import EventSourceResponse

# =====================
# Phase 3/4: INT4 Service for Qwen2-1.5B (Windows + CUDA + RTX2060 6GB)
# - Hard limits: ctx_len / max_new_tokens
# - Concurrency policy: fast reject (429) if busy
# - Endpoints:
#   GET  /health
#   POST /generate        (non-stream JSON)
#   POST /generate_stream (SSE stream)
# =====================

MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"

# Hard limits (carry over Phase 2 policy)
CTX_DEFAULT = 2048
CTX_MAX = 3072

MAX_NEW_TOKENS_DEFAULT = 64
MAX_NEW_TOKENS_MAX = 256

# Concurrency: 1 is the safe default on 6GB VRAM
CONCURRENCY_LIMIT = 1
sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

app = FastAPI(title="Qwen2-1.5B Local Service (INT4, bnb)")

tokenizer = None
model = None


class GenerateReq(BaseModel):
    prompt: str
    ctx_len: Optional[int] = CTX_DEFAULT
    max_new_tokens: Optional[int] = MAX_NEW_TOKENS_DEFAULT


def _encode_tail(text: str, ctx_len: int):
    """
    Tail-truncation:
    Keep the last ctx_len tokens to preserve the newest user question.
    This avoids the "question truncated away" failure mode.
    """
    enc_cpu = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = enc_cpu["input_ids"]
    attn_mask = enc_cpu["attention_mask"]

    if input_ids.shape[1] > ctx_len:
        input_ids = input_ids[:, -ctx_len:]
        attn_mask = attn_mask[:, -ctx_len:]

    return {"input_ids": input_ids.to("cuda"), "attention_mask": attn_mask.to("cuda")}


async def _acquire_or_429():
    """
    Fast reject if semaphore not available quickly (no long queue).
    This improves stability under Windows + long GPU inference.
    """
    try:
        await asyncio.wait_for(sem.acquire(), timeout=0.001)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=429, detail="busy: concurrency limit reached")


def _release():
    try:
        sem.release()
    except ValueError:
        # already released; ignore
        pass


@app.on_event("startup")
def _startup():
    global tokenizer, model
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    # INT4 quantization config (NF4 + double-quant is a good default)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # compute in fp16 on RTX2060
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cuda",
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
    ).eval()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "dtype": "int4",
        "quantization": "bnb-int4-nf4",
        "concurrency_limit": CONCURRENCY_LIMIT,
        "ctx_default": CTX_DEFAULT,
        "ctx_max": CTX_MAX,
        "max_new_tokens_default": MAX_NEW_TOKENS_DEFAULT,
        "max_new_tokens_max": MAX_NEW_TOKENS_MAX,
    }


@app.post("/generate")
async def generate(req: GenerateReq):
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is empty")

    ctx_len = int(req.ctx_len or CTX_DEFAULT)
    max_new_tokens = int(req.max_new_tokens or MAX_NEW_TOKENS_DEFAULT)

    # Hard limits
    ctx_len = max(128, min(ctx_len, CTX_MAX))
    max_new_tokens = max(1, min(max_new_tokens, MAX_NEW_TOKENS_MAX))

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    await _acquire_or_429()
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        enc = _encode_tail(text, ctx_len)

        # TTFT: prefill + first token
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = model.generate(
                **enc,
                max_new_tokens=1,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                use_cache=True,
            )
        torch.cuda.synchronize()
        ttft_s = time.perf_counter() - t0

        # Decode: remaining tokens
        remaining = max(0, max_new_tokens - 1)
        t1 = time.perf_counter()
        with torch.inference_mode():
            out_full = model.generate(
                **enc,
                max_new_tokens=remaining,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                use_cache=True,
            )
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        gen_ids = out_full[0][enc["input_ids"].shape[-1]:]
        answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        new_tokens = int(gen_ids.shape[0])
        decode_time = max(1e-6, (t2 - t1))
        tokens_per_sec = float(new_tokens / decode_time)

        peak_alloc = torch.cuda.max_memory_allocated()
        peak_resv = torch.cuda.max_memory_reserved()

        return {
            "answer": answer,
            "ttft_s": round(ttft_s, 4),
            "new_tokens": new_tokens,
            "tokens_per_sec": round(tokens_per_sec, 2),
            "ctx_len": ctx_len,
            "max_new_tokens": max_new_tokens,
            "peak_vram_alloc_gb": round(peak_alloc / (1024 ** 3), 2),
            "peak_vram_reserved_gb": round(peak_resv / (1024 ** 3), 2),
        }

    except torch.cuda.OutOfMemoryError as e:
        return {
            "error": "OOM",
            "detail": str(e)[:400],
            "ctx_len": ctx_len,
            "max_new_tokens": max_new_tokens,
        }
    finally:
        _release()


@app.post("/generate_stream")
async def generate_stream(req: GenerateReq, request: Request):
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is empty")

    ctx_len = int(req.ctx_len or CTX_DEFAULT)
    max_new_tokens = int(req.max_new_tokens or MAX_NEW_TOKENS_DEFAULT)

    # Hard limits
    ctx_len = max(128, min(ctx_len, CTX_MAX))
    max_new_tokens = max(1, min(max_new_tokens, MAX_NEW_TOKENS_MAX))

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    await _acquire_or_429()
    queue_wait_s = 0.0  # fast-reject mode, no queue

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    enc = _encode_tail(text, ctx_len)

    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

    gen_kwargs = dict(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        use_cache=True,
        streamer=streamer,
    )

    t0 = time.perf_counter()
    first_token_time = None
    approx_chars = 0

    def _run_generate():
        try:
            with torch.inference_mode():
                model.generate(**gen_kwargs)
        except Exception:
            pass

    th = threading.Thread(target=_run_generate, daemon=True)
    th.start()

    async def event_gen():
        nonlocal first_token_time, approx_chars
        try:
            for piece in streamer:
                if await request.is_disconnected():
                    break
                if not piece:
                    continue

                if first_token_time is None:
                    torch.cuda.synchronize()
                    first_token_time = time.perf_counter()

                approx_chars += len(piece)
                yield {"event": "token", "data": piece}

            th.join(timeout=0.2)
            torch.cuda.synchronize()
            t_end = time.perf_counter()

            ttft_s = None if first_token_time is None else (first_token_time - t0)
            total_s = t_end - t0

            peak_alloc = torch.cuda.max_memory_allocated()
            peak_resv = torch.cuda.max_memory_reserved()

            yield {
                "event": "metrics",
                "data": json.dumps(
                    {
                        "ttft_s": None if ttft_s is None else round(ttft_s, 4),
                        "queue_wait_s": round(queue_wait_s, 4),
                        "total_s": round(total_s, 4),
                        "approx_chars": approx_chars,
                        "ctx_len": ctx_len,
                        "max_new_tokens": max_new_tokens,
                        "peak_vram_alloc_gb": round(peak_alloc / (1024 ** 3), 2),
                        "peak_vram_reserved_gb": round(peak_resv / (1024 ** 3), 2),
                    },
                    ensure_ascii=False,
                ),
            }

        except torch.cuda.OutOfMemoryError as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": "OOM", "detail": str(e)[:400]}, ensure_ascii=False),
            }
        except Exception as e:
            yield {
                "event": "error",
                "data": json.dumps({"error": "Exception", "detail": str(e)[:400]}, ensure_ascii=False),
            }
        finally:
            _release()

    return EventSourceResponse(event_gen())
