import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"
CTX_LEN = 512
MAX_NEW_TOKENS = 32

def encode_tail(tokenizer, text: str, ctx_len: int):
    enc_cpu = tokenizer(text, return_tensors="pt", truncation=False)
    input_ids = enc_cpu["input_ids"]
    attn_mask = enc_cpu["attention_mask"]

    if input_ids.shape[1] > ctx_len:
        input_ids = input_ids[:, -ctx_len:]
        attn_mask = attn_mask[:, -ctx_len:]

    return {"input_ids": input_ids.to("cuda"), "attention_mask": attn_mask.to("cuda")}

def main():
    print("=== CONFIG (INT8 smoke) ===")
    print("MODEL_ID=", MODEL_ID)
    print(f"CTX_LEN={CTX_LEN}, MAX_NEW_TOKENS={MAX_NEW_TOKENS}")
    print("torch=", torch.__version__, "cuda=", torch.version.cuda)
    print("===========================")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

    # INT8 load (bitsandbytes)
    t_load0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="cuda",
        load_in_8bit=True,
        low_cpu_mem_usage=True,
    ).eval()
    torch.cuda.synchronize()
    t_load = time.perf_counter() - t_load0

    prompt = "In one sentence, what is KV cache in Transformer inference?"
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    torch.cuda.reset_peak_memory_stats()
    enc = encode_tail(tokenizer, text, CTX_LEN)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )
    torch.cuda.synchronize()
    t_total = time.perf_counter() - t0

    gen_ids = out[0][enc["input_ids"].shape[-1]:]
    answer = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    peak_alloc = torch.cuda.max_memory_allocated()
    peak_resv = torch.cuda.max_memory_reserved()

    print("\n=== RESULT (INT8 smoke) ===")
    print("load_time_s =", round(t_load, 3))
    print("total_time_s =", round(t_total, 3))
    print("new_tokens =", int(gen_ids.shape[0]))
    print("peak_vram_alloc_gb =", round(peak_alloc / (1024 ** 3), 2))
    print("peak_vram_reserved_gb =", round(peak_resv / (1024 ** 3), 2))
    print("\n--- Answer ---")
    print(answer[:400])

if __name__ == "__main__":
    main()
