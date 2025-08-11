import json
import os
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    TrainingArguments, 
    default_data_collator,  
    Trainer
)

# Define model and token
model_name = "meta-llama/Llama-3.2-1B" 
TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # Set your token as environment variable

# Function to load synthetic data from JSON file
def load_synthetic_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    examples = []
    for entry in data:
        name = entry.get("name", "")
        intent_id = entry.get("intent_id", "")  
        utterances = entry.get("utterances", [])
        for utterance in utterances:
            prompt = f"Intent: {name} (ID: {intent_id})\nUtterance: {utterance}\nResponse:"
            examples.append({"text": prompt, "label": intent_id}) 
    return Dataset.from_list(examples)

# Load the dataset
dataset = load_synthetic_data("/home/shtlp_0010/Desktop/Intent Recognition Model Fine-Tuning/data/synthetic_intents.json")
print("Sample dataset entry:", dataset[0])

# Create a label mapping from original labels to integers
def create_label_mapping(dataset):
    unique_labels = list(set([entry["label"] for entry in dataset])) 
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}  
    return label_mapping

# Function to replace label with its corresponding index
def add_labels(example, label_mapping):
    example['label'] = label_mapping[example['label']] 
    return example

# Create and save the label mapping
label_mapping = create_label_mapping(dataset)
mapping_path = "/home/shtlp_0010/Desktop/Intent Recognition Model Fine-Tuning/data/label_mapping.json"
with open(mapping_path, "w", encoding="utf-8") as f:
    json.dump(label_mapping, f, indent=4)
print(f"Label mapping saved to {mapping_path}")

# Update the dataset with new label values
dataset = dataset.map(lambda example: add_labels(example, label_mapping))

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=TOKEN)
tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(example):
    return tokenizer(
        example["text"], 
        truncation=True, 
        padding="max_length", 
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Load the model with the number of labels
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=len(label_mapping)  
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./finetuned_model",         
    per_device_train_batch_size=1,            
    gradient_accumulation_steps=4,           
    warmup_steps=100,                        
    num_train_epochs=3,                      
    learning_rate=5e-5,                       
    save_steps=500,                          
    save_total_limit=2,                      
    logging_steps=50,                        
    fp16=True,                               
    report_to="none"                          
)

data_collator = default_data_collator 

# Create the Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Begin training
trainer.train()

# Save the fine-tuned model and tokenizer
trainer.save_model("./finetuned_model")
tokenizer.save_pretrained("./finetuned_model")
