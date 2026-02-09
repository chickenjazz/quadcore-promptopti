import torch
from transformers import AutoTokenizer
from src.prompts import INSTRUCTION_TEMPLATE
from src.config import OUTPUT_DIR, MODEL_NAME
from peft import PeftModel
from transformers import AutoModelForCausalLM

def load_model():
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base, OUTPUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def optimize_prompt(raw_prompt):
    model, tokenizer = load_model()
    prompt = INSTRUCTION_TEMPLATE.format(raw_prompt=raw_prompt)

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False
    )
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == "__main__":
    inp_prompt = input("Enter your prompt: ")
    optimize_prompt(inp_prompt)
