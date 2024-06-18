from bert_score import score
import json

# Specify the path to your JSON file
file_path = '../dataset/book2qa_sft.json'

# Load data from JSON file
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract 'input' and 'output' fields from each item in the data
inputs = [item['input'] for item in data]
outputs = [item['output'] for item in data]

# Calculate BERT scores (Precision, Recall, F1) for inputs and outputs
P, R, F1 = score(inputs, outputs, model_type="bert-base-chinese", lang="zh", verbose=True)

# Print the F1 scores
print(F1)

# Update JSON data with computed F1 scores
for i in range(len(data)):
    data[i]['bf1'] = float(F1[i])

# Write the updated data back to the original file
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
