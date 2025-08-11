import logging
import json
from datasets import load_dataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class IntentDataset:
    """Handles loading and preprocessing of the intent recognition dataset."""

    def __init__(self, input_file):
        self.input_file = input_file

    def load_data(self):
        """Loads synthetic dataset from JSON file using Hugging Face datasets library."""
        try:
            dataset = load_dataset("json", data_files=self.input_file, split="train")
            logging.info(f"Loaded dataset from {self.input_file}")
            return dataset
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            return None

def expand_dataset(batch):
    """
    Convert each utterance in the list into a separate sample with the same intent label.
    """
    input_texts, output_labels = [], []
    for utterances, label in zip(batch["utterances"], batch["name"]):
        for utt in utterances:
            input_texts.append(utt)
            output_labels.append(label)

    return {"input_text": input_texts, "output_label": output_labels}

def save_to_json(data, output_file):
    """Saves processed dataset to a JSON file."""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logging.info(f"Expanded dataset saved to {output_file}")
    except Exception as e:
        logging.error(f"Error saving dataset: {e}")

# File paths
input_file = "/home/shtlp_0010/Desktop/Intent Recognition Model Fine-Tuning/data/synthetic_intents.json"
output_file = "/home/shtlp_0010/Desktop/Intent Recognition Model Fine-Tuning/data/preprocessed_intents.json"

# Load and expand dataset
dataset_handler = IntentDataset(input_file)
dataset = dataset_handler.load_data()

if dataset:
    expanded_dataset = dataset.map(expand_dataset, batched=True, remove_columns=dataset.column_names)

    # Convert dataset to a list of dictionaries
    expanded_data_list = [{"input_text": item["input_text"], "output_label": item["output_label"]} for item in expanded_dataset]

    # Save expanded dataset to JSON
    save_to_json(expanded_data_list, output_file)

    # Display a few samples for verification
    for i in range(5):  
        print(f"Sample {i+1}:")
        print("Input:", expanded_dataset[i]["input_text"])
        print("Label:", expanded_dataset[i]["output_label"])
        print("-" * 100)
