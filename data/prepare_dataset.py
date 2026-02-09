from datasets import load_dataset
from src.prompts import INSTRUCTION_TEMPLATE
from huggingface_hub import login

def load_mepo_dataset(limit=10000):
    dataset = load_dataset("zixiaozhu/MePO", split="train")
    dataset = dataset.shuffle(seed=42).select(range(limit))

    def format_example(example):
        prompt = INSTRUCTION_TEMPLATE.format(
            raw_prompt=example["raw_prompt"]
        )
        return {
            "prompt": prompt,
            "chosen": example["chosen"],
            "rejected": example["rejected"]
        }

    return dataset.map(format_example)

if __name__ == "__main__":
    ds = load_mepo_dataset(5)
    print(ds[0])
