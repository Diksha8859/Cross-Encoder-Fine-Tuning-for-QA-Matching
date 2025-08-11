import json
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Intent classification prompt template
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

train_valid_test = dataset.train_test_split(train_size=0.70, test_size=0.30, shuffle=True, seed=42)
valid_test = train_valid_test["test"].train_test_split(train_size=0.50, test_size=0.50, shuffle=True, seed=42)

train_dataset = train_valid_test["train"]
valid_dataset = valid_test["train"]
test_dataset = valid_test["test"]

columns_to_remove = train_dataset.column_names

train_dataset = train_dataset.map(formatting_prompts_func, batched=True, remove_columns=columns_to_remove, num_proc=4)
valid_dataset = valid_dataset.map(formatting_prompts_func, batched=True, remove_columns=columns_to_remove, num_proc=4)
test_dataset = test_dataset.map(formatting_prompts_func, batched=True, remove_columns=columns_to_remove, num_proc=4)

def save_dataset(dataset, filename):
    with open(filename, "w") as f:
        for example in dataset:
            json.dump(example, f)
            f.write("\n")

save_dataset(train_dataset, "Test_datasets_model11/train_dataset.jsonl")
save_dataset(valid_dataset, "Test_datasets_model11/valid_dataset.jsonl")
save_dataset(test_dataset, "Test_datasets_model11/test_dataset.jsonl")

print(f"Train size: {len(train_dataset)}, Validation size: {len(valid_dataset)}, Test size: {len(test_dataset)}")
print("Train Sample:", train_dataset[0] if len(train_dataset) > 0 else "Train dataset is empty!")
print("Validation Sample:", valid_dataset[0] if len(valid_dataset) > 0 else "Validation dataset is empty!")
print("Test Sample:", test_dataset[0] if len(test_dataset) > 0 else "Test dataset is empty!")
print("Train Dataset Columns:", train_dataset.column_names)
