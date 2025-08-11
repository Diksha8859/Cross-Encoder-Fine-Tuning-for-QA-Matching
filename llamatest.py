import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "meta-llama/Llama-3.2-1B"

def parse_chat_format(chat_text):
    messages = []
    message_blocks = re.split(r"<\|start_header_id\|>.*?<\|end_header_id\|>", chat_text)
    roles = re.findall(r"<\|start_header_id\|>(.*?)<\|end_header_id\|>", chat_text)
    
    for role, message in zip(roles, message_blocks[1:]):
        cleaned_message = message.strip("\n<|eot_id|>")
        if cleaned_message:
            messages.append({"role": role, "content": cleaned_message})
    
    return messages

def initialize_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
    )
    return tokenizer, model

def generate_response(model, tokenizer, user_input):
    input_ids = tokenizer(user_input, return_tensors="pt").input_ids.to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, max_length=500, do_sample=True, top_p=0.9, temperature=0.7
        )
    
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def chat_with_huggingface():
    print("Hugging Face Chatbot (type 'exit' to quit)")
    tokenizer, model = initialize_model()
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting Chatbot. Goodbye!")
            break
        
        response = generate_response(model, tokenizer, user_input)
        print(f"Bot: {response}")

if __name__ == "__main__":
    chat_with_huggingface()
