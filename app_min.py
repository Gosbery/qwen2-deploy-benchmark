import time
import asyncio
from typing import Optional
import threading
import json

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer


# =====================
# Fixed service policy (Phase 4 baseline)
# =====================
MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"
DTYPE = torch.float16

CTX_DEFAULT = 2048
CTX_MAX = 3072
MAX_NEW_TOKENS_DEFAULT = 64
MAX_NEW_TOKENS_MAX = 256

# Single-concurrency guard (Phase 4)
sem = asyncio.Semaphore(1)

app = FastAPI(title="Qwen2-1.5B Local Service (FP16)")

tokenizer = None
model = None


class GenerateReq(BaseModel):
    prompt: str
    ctx_len: Optional[int] = CTX_DEFAULT
    max_new_tokens: Optional[int] = MAX_NEW_TOKENS_DEFAULT


def _encode_tail(text: str, ctx_len: int):
    """Keep last ctx_len tokens (tail truncation) so the latest user question survives."""
    enc_cpu = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = enc_cpu["input_ids"]
    attn_mask = enc_cpu["attention_mask"]

    if input_ids.shape[1] > ctx_len:
        input_ids = input_ids[:, -ctx_len:]
        attn_mask = attn_mask[:, -ctx_len:]

    return {"input_ids": input_ids.to("cuda"), "attention_mask": attn_mask.to("cuda")}


@app.on_event("startup")
def _startup():
    global tokenizer, model
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map="cuda",
        low_cpu_mem_usage=True,
    ).eval()


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "dtype": "fp16"}


@app.post("/generate")
async def generate(req: GenerateReq):
    """Non-streaming endpoint: returns JSON with ttft + throughput."""
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is empty")

    ctx_len = int(req.ctx_len or CTX_DEFAULT)
    max_new_tokens = int(req.max_new_tokens or MAX_NEW_TOKENS_DEFAULT)

    ctx_len = max(128, min(ctx_len, CTX_MAX))
    max_new_tokens = max(1, min(max_new_tokens, MAX_NEW_TOKENS_MAX))

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    async with sem:
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


@app.post("/generate_stream")
async def generate_stream(req: GenerateReq):
    if tokenizer is None or model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is empty")

    ctx_len = int(req.ctx_len or CTX_DEFAULT)
    max_new_tokens = int(req.max_new_tokens or MAX_NEW_TOKENS_DEFAULT)

    # Enforce hard limits (from Phase 2)
    ctx_len = max(128, min(ctx_len, CTX_MAX))
    max_new_tokens = max(1, min(max_new_tokens, MAX_NEW_TOKENS_MAX))

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # queue wait should measure time until we REALLY acquire the semaphore
    t_queue0 = time.perf_counter()

    async def event_gen():
        # IMPORTANT: semaphore must be held for the ENTIRE streaming lifecycle
        async with sem:
            queue_wait_s = time.perf_counter() - t_queue0

            try:
                import json
                import threading

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
                new_chars = 0

                def _run_generate():
                    with torch.inference_mode():
                        model.generate(**gen_kwargs)

                th = threading.Thread(target=_run_generate, daemon=True)
                th.start()

                started = False
                buffer = ""

                for piece in streamer:
                    if not piece:
                        continue

                    # TTFT measured at first piece arriving from streamer
                    if first_token_time is None:
                        torch.cuda.synchronize()
                        first_token_time = time.perf_counter()

                    buffer += piece

                    # Gate: start streaming after assistant content begins
                    if not started:
                        lower_buf = buffer.lower()
                        if "assistant" in lower_buf:
                            idx = lower_buf.rfind("assistant")
                            buffer = buffer[idx + len("assistant"):]
                            buffer = buffer.lstrip(": \r\n\t")
                            started = True
                        else:
                            continue

                    if buffer:
                        new_chars += len(buffer)
                        yield {"event": "token", "data": buffer}
                        buffer = ""

                th.join(timeout=0.1)
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
                            "approx_chars": new_chars,
                            "ctx_len": ctx_len,
                            "max_new_tokens": max_new_tokens,
                            "peak_vram_alloc_gb": round(peak_alloc / (1024 ** 3), 2),
                            "peak_vram_reserved_gb": round(peak_resv / (1024 ** 3), 2),
                        },
                        ensure_ascii=False,
                    ),
                }

            except torch.cuda.OutOfMemoryError as e:
                yield {"event": "error", "data": f"OOM: {str(e)[:400]}"}
            except Exception as e:
                yield {"event": "error", "data": f"Exception: {str(e)[:400]}"}

    return EventSourceResponse(event_gen())

import threading
import requests

class SelfTestReq(BaseModel):
    ctx_len: Optional[int] = 2048
    long_new_tokens: Optional[int] = 256
    short_new_tokens: Optional[int] = 32

@app.post("/selftest_queue")
def selftest_queue(req: SelfTestReq):
    """
    Deterministic concurrency test:
    - Fire a long streaming request A in a background thread (holds semaphore ~20-30s)
    - After 1s, fire a short streaming request B in main thread
    - Return BOTH metrics by parsing the final "event: metrics" JSON from SSE streams
    """
    ctx_len = int(req.ctx_len or 2048)
    long_new = int(req.long_new_tokens or 256)
    short_new = int(req.short_new_tokens or 32)

    url = "http://127.0.0.1:8000/generate_stream"
    headers = {"Content-Type": "application/json"}

    long_body = {
        "prompt": "Explain KV cache in Transformer inference in extreme detail. Write at least 10 sentences.",
        "ctx_len": ctx_len,
        "max_new_tokens": long_new,
    }
    short_body = {
        "prompt": "What is KV cache? 1 sentence.",
        "ctx_len": ctx_len,
        "max_new_tokens": short_new,
    }

    # helper: read SSE and capture final metrics json line
    def _read_metrics(body):
        r = requests.post(url, headers=headers, json=body, stream=True, timeout=300)
        r.raise_for_status()
        metrics_json = None
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.strip()
            if line.startswith("data:"):
                payload = line[len("data:"):].strip()
                # metrics is a JSON object
                if payload.startswith("{") and '"queue_wait_s"' in payload:
                    metrics_json = payload
        return metrics_json

    out = {"A_metrics": None, "B_metrics": None}

    def _run_A():
        try:
            out["A_metrics"] = _read_metrics(long_body)
        except Exception as e:
            out["A_metrics"] = f"ERROR: {str(e)[:200]}"

    th = threading.Thread(target=_run_A, daemon=True)
    th.start()

    time.sleep(1.0)  # ensure A acquires semaphore first

    try:
        out["B_metrics"] = _read_metrics(short_body)
    except Exception as e:
        out["B_metrics"] = f"ERROR: {str(e)[:200]}"

    th.join(timeout=1.0)
    return out

