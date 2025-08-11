import json
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

_intent_prompt = """Intent Classification Data
### Utterance: {}

### Intent:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    if "utterances" not in examples or "name" not in examples:
        raise ValueError("Missing required keys in dataset: 'utterances' and 'name'")

    utterances_batch = examples["utterances"]
    intents_batch = examples["name"]

    formatted_texts = [
        _intent_prompt.format(utterance, intent) + (EOS_TOKEN or "")
        for utterances, intent in zip(utterances_batch, intents_batch)
        for utterance in (utterances if isinstance(utterances, list) else [utterances])
    ]
    return {"text": formatted_texts}

dataset = load_dataset("json", data_files="/home/shtlp_0010/Desktop/Intent Recognition Model Fine-Tuning/data/my_data.json", split="train")

# dataset = dataset.train_test_split(train_size=0.8, shuffle=True, seed=42)["train"]

dataset = dataset.map(formatting_prompts_func, batched=True, remove_columns=dataset.column_names)

output_file = "Test_dataset_model8/formatted_dataset.jsonl"
with open(output_file, "w") as f:
    for example in dataset:
        json.dump(example, f)
        f.write("\n")

print(f"Dataset saved to {output_file}")
print(dataset[0] if len(dataset) > 0 else "Dataset is empty!")
