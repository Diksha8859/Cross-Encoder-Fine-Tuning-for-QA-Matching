import os
import json
import time
import random
import google.generativeai as genai
from tqdm import tqdm

class SyntheticIntentGenerator:
    """Class to generate synthetic intent utterances using the Gemini API."""

    PROMPT_TEMPLATES = [
    "Generate 5 clear and natural user utterances for the intent '{intent_name}'. Ensure variation in sentence structure, word choice, and tone while keeping them concise and free of unnecessary characters. Use the following examples as inspiration: {examples}.",
    "Provide 5 distinct ways a user might express the intent '{intent_name}'. Use synonyms, varied sentence structures, and natural phrasing while ensuring clarity and brevity. Use these examples for reference: {examples}.",
    "Rephrase the intent '{intent_name}' into 5 clean and distinct utterances. Each should differ in formality, phrasing, and structure while preserving meaning. Ensure they are free of extra characters. Reference examples: {examples}.",
    "List 5 alternative ways a user might convey the intent '{intent_name}'. Modify sentence structure, word order, and vocabulary while ensuring readability and removing unnecessary elements. Use these examples as guidance: {examples}.",
    "Generate 5 natural-sounding user utterances for the intent '{intent_name}', ensuring clarity and variation in structure, word choice, and phrasing. Keep them concise and free of extraneous characters. Take inspiration from these examples: {examples}."
]


    def __init__(self, api_key: str, input_file: str, output_file: str):
        genai.configure(api_key=api_key)
        self.input_file = input_file
        self.output_file = output_file

    def load_intents(self) -> list:
        """Load intents from a JSON file."""
        with open(self.input_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def generate_synthetic_data(self, intent_name: str, utterances: list) -> list:
        """Generate diverse synthetic utterances using Gemini API."""
        time.sleep(5)  
        example_text = "; ".join(utterances)
        prompt = random.choice(self.PROMPT_TEMPLATES).format(intent_name=intent_name, examples=example_text)
        
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)

        generated_text = response.text.strip() if response.text else ""
        
        if not generated_text:
            return []
        
        return [sentence.strip() for sentence in generated_text.split("\n") if sentence.strip()][:5]

    def create_synthetic_dataset(self, intents_data: list) -> list:
        """Generate a dataset with synthetic utterances."""
        synthetic_dataset = []
        for item in tqdm(intents_data, desc="Generating Synthetic Data"):
            generated_utterances = self.generate_synthetic_data(item["name"], item["utterances"])
            
            if generated_utterances:
                synthetic_dataset.append({
                    "name": item["name"],
                    "intent_id": item["intent_id"],
                    "utterances": generated_utterances
                })
        return synthetic_dataset

    def save_dataset(self, data: list):
        """Save dataset to JSON file."""
        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"Synthetic dataset saved to {self.output_file}")

    def run(self):
        """Execute the synthetic data generation pipeline."""
        intents_data = self.load_intents()
        synthetic_dataset = self.create_synthetic_dataset(intents_data)
        self.save_dataset(synthetic_dataset)

if __name__ == "__main__":
    BASE_DIR = "/home/shtlp_0010/Desktop/Intent Recognition Model Fine-Tuning/data"
    INPUT_FILE = os.path.join(BASE_DIR, "intents.json")
    OUTPUT_FILE = os.path.join(BASE_DIR, "synthetic_intents.json")
    API_KEY = os.getenv("GOOGLE_API_KEY")  # Set your API key as environment variable

    if not API_KEY:
        raise ValueError("Please set GOOGLE_API_KEY environment variable")

    generator = SyntheticIntentGenerator(api_key=API_KEY, input_file=INPUT_FILE, output_file=OUTPUT_FILE)
    generator.run()
