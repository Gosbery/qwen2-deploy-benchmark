import time
import threading
import traceback

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

# ====== Hard constraints (do not change for baseline) ======
MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"
CTX_LEN = 512
MAX_NEW_TOKENS = 64
CONCURRENCY = 1
# ===========================================================

def gb(x: int) -> str:
    return f"{x / (1024**3):.2f} GB"

def main():
    assert CONCURRENCY == 1, "Baseline must be concurrency=1"

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    print("=== CONFIG ===")
    print(f"MODEL_ID={MODEL_ID}")
    print(f"CTX_LEN={CTX_LEN}, MAX_NEW_TOKENS={MAX_NEW_TOKENS}, CONCURRENCY={CONCURRENCY}")
    print(f"GPU={torch.cuda.get_device_name(0)}")
    print(f"torch={torch.__version__}, torch_cuda_runtime={torch.version.cuda}")
    print("=============\n")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
        low_cpu_mem_usage=True,
    )
    model.eval()

    prompt = "Briefly explain what KV cache is in transformer inference."
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=CTX_LEN).to("cuda")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # warmup
    with torch.inference_mode():
        _ = model.generate(**enc, max_new_tokens=8, do_sample=False, use_cache=True)
    torch.cuda.synchronize()

    gen_kwargs = dict(
        **enc,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        use_cache=True,
        streamer=streamer,
    )

    print("=== BENCHMARK (streaming) ===")
    t0 = time.perf_counter()
    ttft = None
    chunks = []

    def _gen():
        with torch.inference_mode():
            model.generate(**gen_kwargs)

    th = threading.Thread(target=_gen, daemon=True)
    th.start()

    try:
        for ch in streamer:
            if ttft is None:
                ttft = time.perf_counter() - t0
            chunks.append(ch)
        th.join()
    except Exception:
        print("Streaming exception:")
        traceback.print_exc()

    t1 = time.perf_counter()
    torch.cuda.synchronize()

    out_text = "".join(chunks).strip()
    out_ids = tokenizer(out_text, return_tensors="pt").input_ids
    new_tokens = int(out_ids.shape[-1])
    total = t1 - t0
    tps = (new_tokens / total) if total > 0 else None

    peak_alloc = torch.cuda.max_memory_allocated()
    peak_resv = torch.cuda.max_memory_reserved()

    print("\n=== RESULT ===")
    print(f"TTFT = {ttft:.3f} s" if ttft is not None else "TTFT = N/A")
    print(f"Total time = {total:.3f} s")
    print(f"New tokens = {new_tokens}")
    print(f"Tokens/sec = {tps:.2f}" if tps is not None else "Tokens/sec = N/A")
    print(f"Peak VRAM allocated = {gb(peak_alloc)}")
    print(f"Peak VRAM reserved  = {gb(peak_resv)}")
    print("\n--- Output (first 400 chars) ---")
    print(out_text[:400])
    print("\n[OK] FP16 baseline done.")

if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError as e:
        print("\n[OOM] torch.cuda.OutOfMemoryError")
        print(f"CTX_LEN={CTX_LEN}, MAX_NEW_TOKENS={MAX_NEW_TOKENS}, CONCURRENCY={CONCURRENCY}")
        print(str(e))
    except Exception as e:
        print("\n[ERROR]")
        print(str(e))
        traceback.print_exc()
