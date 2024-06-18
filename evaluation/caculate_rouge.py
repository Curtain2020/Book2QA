from rouge_chinese import Rouge  # Assuming Rouge is properly imported
import jieba
import json
import os

# Specify the path to your JSON file
file_path = '../dataset/book2qa_sft.json'

# Load data from JSON file
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Initialize Rouge object
rouge = Rouge()

# Process each item in the data
for item in data:
    question = item['input']
    answer = item['output']
    paragraph = item['paragraph']
    
    # Tokenize using jieba
    hypothesis = ' '.join(jieba.cut(question))
    reference = ' '.join(jieba.cut(paragraph))
    
    # Calculate Rouge scores
    scores = rouge.get_scores(hypothesis, reference)
    
    # Add Rouge scores to the item
    item['pq-rouge'] = scores

# Write the updated data back to the original file
with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
