import json
import re
from tqdm import tqdm
from llama_cpp import Llama

MODEL_PATH = "/home/shtlp_0010/Desktop/Intent Recognition Model Fine-Tuning/gguf_models/ggufmodel20/unsloth.Q4_K_M.gguf"
llm = Llama(model_path=MODEL_PATH, n_ctx=2048)

def extract_utterance_intent(text):
    """
    Extracts utterance and intent from the given text using regex.
    """
    match = re.search(r"### Utterance:\s*(.*?)\s*\n+### Intent:\s*(.*)", text, re.DOTALL)
    if match:
        utterance = match.group(1).strip().lower()
        intent = match.group(2).strip().lower()
        return utterance, intent
    return None, None

evaluation_file = "/home/shtlp_0010/Desktop/Intent Recognition Model Fine-Tuning/Test_datasets_model11/valid_dataset.jsonl"
test_data = []

with open(evaluation_file, "r", encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line)
            utterance, intent = extract_utterance_intent(data["text"])
            if utterance and intent:
                test_data.append((utterance, intent))
        except json.JSONDecodeError:
            print("Skipping malformed JSON entry.")

def clean_predicted_intent(intent):
    intent = intent.strip().lower()
    intent = re.sub(r"[^a-z0-9_]", "_", intent)  
    intent = re.sub(r"_+", "_", intent) 
    intent = re.sub(r"_error|_issue|_problem", "_issue", intent) 
    intent = intent.strip("_") 

    if len(intent) < 3 or intent.count("_") > 4:
        return "unknown"
    
    return intent


def predict_intent(utterance):
    prompt = (
        "You are a high-accuracy intent classifier.\n"
        "Below are examples of user utterances and their correct intent labels:\n\n"
        "### Utterance: Why was I charged an incorrect exchange rate?\n"
        "### Intent: card_payment_wrong_exchange_rate\n\n"
        "### Utterance: My card shows a different exchange rate than expected.\n"
        "### Intent: card_payment_wrong_exchange_rate\n\n"
        "### Utterance: I noticed an extra charge on my statement.\n"
        "### Intent: extra_charge_on_statement\n\n"
        "Now classify the following:\n\n"
        f"### Utterance: {utterance}\n\n### Intent:\n"
    )

    response = llm(prompt, max_tokens=10, echo=False, temperature=0.1, top_p=0.8, top_k=40)

    if isinstance(response, dict) and "choices" in response:
        predicted_intent = response["choices"][0]["text"].strip().split("\n")[0]
        return clean_predicted_intent(predicted_intent)

    return "unknown"



def exact_match(predicted, actual):
    """Strict exact match after cleaning both labels."""
    return clean_predicted_intent(predicted) == clean_predicted_intent(actual)

# Model Evaluation
correct_predictions = 0
total_samples = len(test_data)

print("\nEvaluating model on test dataset...\n")

for utterance, true_intent in tqdm(test_data, desc="Processing", unit="sample"):
    predicted_intent = predict_intent(utterance)
    if exact_match(predicted_intent, true_intent):
        correct_predictions += 1

percentage_match = (correct_predictions / total_samples * 100) if total_samples > 0 else 0

print("\nEvaluation Results:")
print(f"Total Queries: {total_samples}")
print(f"Matched Queries: {correct_predictions}")
print(f"Match Percentage: {percentage_match:.2f}%")

print("\nEnter utterances to predict intents (Press Enter without input to exit):")
while True:
    user_utterance = input("\nEnter an utterance: ").strip()

    if user_utterance == "":
        print("Exiting...")
        break

    print(f"Predicted Intent: {predict_intent(user_utterance)}")
