from model_setup import load_model
from data_preprocessing import load_and_expand_dataset
from lookup_utils import build_lookup_table, find_label
from train import train_model, save_model

def main():
    # Load model
    model, tokenizer = load_model()

    # Load dataset
    dataset = load_and_expand_dataset("/content/synthetic_intents.json")

    # Build lookup table
    lookup_table = build_lookup_table(dataset)

    # Find a label
    utterance = "When will I get my physical card in the mail?"
    label = find_label(utterance, lookup_table)
    print(f"Label for '{utterance}': {label}")

    # Train model
    trainer_stats = train_model(model, tokenizer, dataset)

    # Save model
    save_model(model, tokenizer)

if __name__ == "__main__":
    main()
