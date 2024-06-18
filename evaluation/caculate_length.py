from bert_score import score
import json
import os

# Specify the path to your JSON file
file_path = '../dataset/book2qa_sft.json'

# Load data from JSON file
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Calculate lengths of 'input' and 'output' and update each item in data
for item in data:
    item['input_length'] = len(item['input'])
    item['output_length'] = len(item['output'])

# Write the updated data back to the original file
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
