import json

# Define input and output file paths
input_file = "/home/shtlp_0010/Desktop/Intent Recognition Model Fine-Tuning/data/my_data.json"  # Input JSON file
output_file = "extracted_intents.json"  # Output JSON file

# Read the input JSON file
with open(input_file, "r", encoding="utf-8") as file:
    json_data = json.load(file)  # Load JSON data as a list

# List to store extracted intents
data = []

# Process each entry in the JSON file
for entry in json_data:
    intent = entry.get("name")  # Extract intent name
    if intent:
        data.append({"intent": intent})

# Write the extracted intents to a JSON file
with open(output_file, "w", encoding="utf-8") as output:
    json.dump(data, output, indent=4, ensure_ascii=False)

print(f"Extracted intents saved to {output_file}")