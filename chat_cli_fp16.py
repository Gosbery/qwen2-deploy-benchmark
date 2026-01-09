import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"

# Hard limits (for 6GB VRAM stability)
CTX_LEN = 512
MAX_NEW_TOKENS = 128

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cuda",
        low_cpu_mem_usage=True,
    ).eval()

    system = "You are a helpful assistant. Answer concisely."
    history = [{"role": "system", "content": system}]

    print("\n[Qwen2-1.5B CLI] Type your message. Type 'exit' to quit.\n")

    while True:
        user = input("You: ").strip()
        if user.lower() in ("exit", "quit"):
            break
        if not user:
            continue

        history.append({"role": "user", "content": user})

        # Build prompt with chat template, and truncate to CTX_LEN tokens
        prompt_text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=CTX_LEN).to("cuda")

        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                use_cache=True,
            )

        # Only decode newly generated part
        gen_ids = out[0][enc["input_ids"].shape[-1]:]
        assistant = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        print(f"\nQwen: {assistant}\n")

        history.append({"role": "assistant", "content": assistant})

if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError as e:
        print("\n[OOM] Reduce CTX_LEN / MAX_NEW_TOKENS and retry.")
        print(str(e))
