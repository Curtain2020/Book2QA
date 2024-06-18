import os
import json
import jieba
import collections
import math

def calculate_entropy(word_list, word_set):
    # Function to calculate entropy based on a list of words and a set of all possible words
    
    # Count occurrences of each word in the word list
    word_count = collections.Counter(word_list)
    
    # Total number of words in the list
    total_words = len(word_list)
    
    # Calculate probabilities of each word
    word_probabilities = {word: count / total_words for word, count in word_count.items()}
    
    # Calculate entropy
    entropy = 0
    for prob in word_probabilities.values():
        if prob != 0:
            entropy -= prob * math.log(prob, 2)
    return entropy

def main(folder_path):
    # Main function to process data and calculate entropy
    
    # Specify the path to your JSON file
    file_path = '../dataset/book2qa_sft.json'
    
    # Load data from JSON file
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    all_input_words = set()   # Set to store all unique input words
    all_output_words = set()  # Set to store all unique output words
    
    # Collect all unique input and output words
    for item in data:
        input_text = item['input']
        input_words = set(jieba.cut(input_text))
        all_input_words.update(input_words)
        
    for item in data:
        output_text = item['output']
        output_words = set(jieba.cut(output_text))
        all_output_words.update(output_words)
    
    # Calculate and update entropy for each input-output pair
    for item in data:
        input_text = item['input']
        output_text = item['output']
        
        # Tokenize input and output texts
        input_words = list(jieba.cut(input_text))
        output_words = list(jieba.cut(output_text))
        
        # Calculate entropy for input and output
        input_entropy = calculate_entropy(input_words, all_input_words)
        output_entropy = calculate_entropy(output_words, all_output_words)
        
        # Update item with calculated entropy values
        item['input_entropy'] = input_entropy
        item['output_entropy'] = output_entropy
    
    # Write the updated data back to the original JSON file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# Example usage of main function
if __name__ == "__main__":
    folder_path = '.'  # Replace with your actual folder path if needed
    main(folder_path)
