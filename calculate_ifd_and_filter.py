import json
import torch
from sklearn.metrics.pairwise import euclidean_distances
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from collections import defaultdict
from tqdm import tqdm
import os

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-7B-Chat", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan2-7B-Chat", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.eval()
model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan2-7B-Chat")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Function to get perplexity and embedding for the whole text
def get_perplexity_and_embedding_whole_text(tokenizer, model, text, max_length):
    try:
        input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        with torch.no_grad(): 
            outputs = model(input_ids, labels=input_ids.contiguous())
        loss = outputs.loss
        perplexity = torch.exp(loss)
        return perplexity.to('cpu').item(), loss.to('cpu').item()
    except:
        return 0, 0

# Function to get perplexity and embedding for part of the text
def get_perplexity_and_embedding_part_text(tokenizer, model, text, target_span, max_length):
    try:
        input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        start_index = text.rfind(target_span)
        start_token = len(tokenizer.encode(text[:start_index]))
        labels = input_ids.clone()
        labels[0, :start_token] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        perplexity = torch.exp(loss)
        return perplexity.to('cpu').item(), loss.to('cpu').item()
    except:
        return 0, 0

# Function to generate an answer using the model
def generate_answer(instruction):
    messages = [{"role": "user", "content": instruction}]
    generated_text = model.chat(tokenizer, messages)
    if "[答案]:" in generated_text:
        answer_index = generated_text.index("[答案]:") + len("[答案]:")
        answer = generated_text[answer_index:].strip()
    else:
        answer = generated_text
    return answer

# Function to calculate the IFD
def calculate_ifd(query):
    output_i = generate_answer(query)
    whole_text = query + "[答案]:" + output_i
    instruct_i = query

    max_length = 2048
    instruct_i_input_ids = tokenizer.encode(instruct_i, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    instruct_i_len = instruct_i_input_ids.shape[1]

    ppl_out_alone, loss_out_alone = get_perplexity_and_embedding_whole_text(tokenizer, model, output_i, max_length - instruct_i_len + 1)
    ppl_out_condition, loss_out_condition = get_perplexity_and_embedding_part_text(tokenizer, model, whole_text, output_i, max_length)
    ifd_ppl = ppl_out_condition / ppl_out_alone if ppl_out_alone != 0 else 0
    return ifd_ppl

# Load merged_question.json file
with open('../dataset/merged_question.json', 'r') as infile:
    merged_data = json.load(infile)

# Iterate through each question and calculate the IFD value
for group in tqdm(merged_data['merged_data']):
    for item in group['group']:
        for question_info in item['questions']:
            question_info['ifd'] = calculate_ifd(question_info['question'])

# Create a dictionary to store the highest IFD value question for each cluster
cluster_max_ifd = {}

# Iterate through the data items in "merged_data"
for group in merged_data['merged_data']:
    for item in group['group']:
        for question_info in item['questions']:
            cluster = question_info["cluster"]
            ifd = question_info["ifd"]
            file_name = question_info["file_name"]

            # Update the dictionary if the current cluster is not in it or the current IFD value is higher than the existing one
            if cluster not in cluster_max_ifd or ifd > cluster_max_ifd[cluster]["ifd"]:
                cluster_max_ifd[cluster] = {"ifd": ifd, "question": question_info["question"], "file_name": question_info["file_name"]}

# Update the "group" questions to retain only the highest IFD value question for each cluster
for group in merged_data['merged_data']:
    group['group'] = [{"question": v["question"], "ifd": v["ifd"], "file_name": v["file_name"], "cluster": k} for k, v in cluster_max_ifd.items()]

# Write the final result to ../dataset/generated_questions.json
with open('../dataset/generated_questions.json', 'w') as outfile:
    json.dump(merged_data, outfile, indent=4, ensure_ascii=False)
