import json
import re
import torch
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

    with open("tasks.json") as f:
        tasks = json.load(f)

    results = {}

    for mode, prompt_fn in PROMPTS.items():
        correct = 0
        violations = 0
        token_counts = []

        for task in tqdm(tasks, desc=f"Mode: {mode}"):
            prompt = prompt_fn(task["question"])
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False
                )

            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            answer = extract_answer(decoded)

            if answer == task["answer"]:
                correct += 1

            if structure_violation(decoded, mode):
                violations += 1

            token_counts.append(len(output[0]) - len(inputs["input_ids"][0]))

        results[mode] = {
            "accuracy": correct / len(tasks),
            "structure_violation_rate": violations / len(tasks),
            "avg_output_tokens": sum(token_counts) / len(token_counts),
        }

    print("\n=== RESULTS ===")
    for mode, stats in results.items():
        print(f"\n[{mode}]")
        for k, v in stats.items():
            print(f"{k}: {v:.3f}")

if __name__ == "__main__":
    main()
