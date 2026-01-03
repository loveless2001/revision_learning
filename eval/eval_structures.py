import json
import re
import torch
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompts import xml_prompt, natural_prompt, glyph_prompt

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # change if needed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 512

PROMPTS = {
    "xml": xml_prompt,
    "natural": natural_prompt,
    "glyph": glyph_prompt,
}

def extract_answer(text):
    nums = re.findall(r"-?\d+\.?\d*", text)
    return nums[-1] if nums else None

def structure_violation(text, mode):
    if mode == "xml":
        return not all(tag in text for tag in ["<guideline>", "<plan>", "<step>", "<takeaway>"])
    if mode == "natural":
        return not all(k in text for k in ["Guideline", "Plan", "Step", "Takeaway"])
    if mode == "glyph":
        return not all(g in text for g in ["🜞", "🜆", "🜂", "🜃"])
    return True

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "tasks.json")) as f:
        tasks = json.load(f)

    results = {}

    for mode, prompt_fn in PROMPTS.items():
        correct = 0
        violations = 0
        total_reasoning_tokens = 0
        total_answer_tokens = 0

        # Define markers for each mode
        marker_map = {
            "glyph": "🜃",
            "xml": "<takeaway>",
            "natural": "Takeaway:"
        }
        marker = marker_map.get(mode)

        for task in tqdm(tasks, desc=f"Mode: {mode}"):
            prompt = prompt_fn(task["question"])
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False
                )
            
            # Decode only variables tokens
            gen_tokens = output[0][input_len:]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            
            # Split by marker
            reasoning_part = gen_text
            answer_part = ""
            
            if marker and marker in gen_text:
                # User logic: extract after final marker
                # text.split(marker)[-1] gets the last part
                parts = gen_text.split(marker)
                answer_part = parts[-1] 
                # Reasoning is everything before the final answer part (including the marker or not?)
                # To be precise with "tokens before final answer", we take everything up to the split.
                reasoning_part = gen_text[:-(len(answer_part) + len(marker))]
                # Note: This crude slicing assumes clean reconstruction. 
                # Safer: reasoning_part = marker.join(parts[:-1])

            # Extract answer from the appropriate part
            # If marker found, search answer_part. Else search full gen_text (fallback or failure)
            search_text = answer_part if (marker and marker in gen_text) else gen_text
            extracted = extract_answer(search_text)

            if extracted == task["answer"]:
                correct += 1

            # Determine violation on generated text
            if structure_violation(gen_text, mode):
                violations += 1
            
            # Count tokens
            # We use the tokenizer to count tokens in the substrings
            # add_special_tokens=False purely counts content
            r_tokens = len(tokenizer.encode(reasoning_part, add_special_tokens=False))
            a_tokens = len(tokenizer.encode(answer_part, add_special_tokens=False))
            
            total_reasoning_tokens += r_tokens
            total_answer_tokens += a_tokens

        results[mode] = {
            "accuracy": correct / len(tasks),
            "structure_violation_rate": violations / len(tasks),
            "avg_reasoning_tokens": total_reasoning_tokens / len(tasks),
            "avg_answer_tokens": total_answer_tokens / len(tasks),
            "avg_total_tokens": (total_reasoning_tokens + total_answer_tokens) / len(tasks),
        }

    print("\n=== RESULTS ===")
    for mode, stats in results.items():
        print(f"\n[{mode}]")
        for k, v in stats.items():
            print(f"{k}: {v:.3f}")

if __name__ == "__main__":
    main()
