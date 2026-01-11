# run_int4_smoke.py
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# =========================
# Phase 3: INT4 smoke test
# =========================
MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"
CTX_LEN = 512
MAX_NEW_TOKENS = 32

PROMPT = "In one sentence, what is KV cache in Transformer inference?"

def _encode_tail(tokenizer, text: str, ctx_len: int):
    """
    Tail truncation: keep last ctx_len tokens to avoid the "question truncated away" issue.
    """
    enc_cpu = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = enc_cpu["input_ids"]
    attn_mask = enc_cpu["attention_mask"]

    if input_ids.shape[1] > ctx_len:
        input_ids = input_ids[:, -ctx_len:]
        attn_mask = attn_mask[:, -ctx_len:]

    return {"input_ids": input_ids.to("cuda"), "attention_mask": attn_mask.to("cuda")}


def main():
    print("=== CONFIG (INT4 smoke) ===")
    print("MODEL_ID=", MODEL_ID)
    print(f"CTX_LEN={CTX_LEN}, MAX_NEW_TOKENS={MAX_NEW_TOKENS}")
    print("torch=", torch.__version__, "cuda=", torch.version.cuda)
    print("===========================")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    # INT4 quant config (bnb 4bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # common choice: nf4
        bnb_4bit_use_double_quant=True,      # improves accuracy at small overhead
        bnb_4bit_compute_dtype=torch.float16 # fp16 compute on RTX 2060
    )

    # Load tokenizer + model
    t_load0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cuda",
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
    ).eval()
    torch.cuda.synchronize()
    load_time_s = time.perf_counter() - t_load0

    # Prepare prompt (chat template)
    messages = [{"role": "user", "content": PROMPT}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Run one generation
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    try:
        enc = _encode_tail(tokenizer, text, CTX_LEN)

        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                use_cache=True,
            )

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        gen_ids = out[0][enc["input_ids"].shape[-1]:]
        answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        new_tokens = int(gen_ids.shape[0])
        total_time_s = t1 - t0
        tokens_per_sec = float(new_tokens / max(1e-6, total_time_s))

        peak_alloc = torch.cuda.max_memory_allocated()
        peak_resv = torch.cuda.max_memory_reserved()

        print("\n=== RESULT (INT4 smoke) ===")
        print(f"load_time_s = {load_time_s:.3f}")
        print(f"total_time_s = {total_time_s:.3f}")
        print(f"new_tokens = {new_tokens}")
        print(f"tokens_per_sec = {tokens_per_sec:.2f}")
        print(f"peak_vram_alloc_gb = {peak_alloc / (1024**3):.2f}")
        print(f"peak_vram_reserved_gb = {peak_resv / (1024**3):.2f}")
        print("\n--- Answer ---")
        print(answer)

    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.synchronize()
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_resv = torch.cuda.max_memory_reserved()

        print("\n=== RESULT (INT4 smoke) ===")
        print("OOM = Yes")
        print("detail =", str(e)[:400])
        print(f"peak_vram_alloc_gb = {peak_alloc / (1024**3):.2f}")
        print(f"peak_vram_reserved_gb = {peak_resv / (1024**3):.2f}")


if __name__ == "__main__":
    main()
