import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation_utils import GenerationConfig
from collections import defaultdict
from tqdm import tqdm
import math

# Load pretrained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("Baichuan2-7B-Chat", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Baichuan2-7B-Chat", device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), torch_dtype=torch.bfloat16, trust_remote_code=True)
model.eval()
model.generation_config = GenerationConfig.from_pretrained("Baichuan2-7B-Chat")

# Function to calculate perplexity and embedding for the entire text
def get_perplexity_and_embedding_whole_text(tokenizer, model, text, max_length):
    try:
        input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids.contiguous())
        loss = outputs.loss
        perplexity = torch.exp(loss)
        return perplexity.to('cpu').item(), loss.to('cpu').item()
    except Exception as e:
        print(f"Error in get_perplexity_and_embedding_whole_text: {str(e)}")
        return 0, 0

# Function to calculate perplexity and embedding for a part of the text
def get_perplexity_and_embedding_part_text(tokenizer, model, text, target_span, max_length):
    try:
        input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
        start_index = text.rfind(target_span)
        start_token = len(tokenizer.encode(text[:start_index]))
        labels = input_ids.clone()
        labels[0, :start_token] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        perplexity = torch.exp(loss)
        return perplexity.to('cpu').item(), loss.to('cpu').item()
    except Exception as e:
        print(f"Error in get_perplexity_and_embedding_part_text: {str(e)}")
        return 0, 0

# Function to generate an answer based on a given question
def generate_answer(question):
    input_text = f"user: {question}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        output = model.generate(input_ids, max_length=100)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    if "[答案]:" in generated_text:
        answer_index = generated_text.index("[答案]:") + len("[答案]:")
        answer = generated_text[answer_index:].strip()
    else:
        answer = generated_text
    return answer

# Function to write clusters to a file
def write_clusters_to_file(id, clusters, output_file):
    with open(output_file, 'a', encoding='utf-8') as f:
        json.dump({id: clusters[id]}, f, ensure_ascii=False, indent=4)
        f.write(',')
        f.write('\n')

# Function to calculate r-IFD (Reverse Information Flow Discrepancy)
def calculate_rifd(question, answer):
    max_length = 2048
    instruct_i_reverse = f"以下是对一个[问题]的[回答]，请猜测给定[回答]的相应[问题]。[回答]: {answer}\n生成上述[回答]的[问题]。[问题]: {question}"
    instruct_i_reverse_input_ids = tokenizer.encode(instruct_i_reverse, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    instruct_i_reverse_len = instruct_i_reverse_input_ids.shape[1]

    ppl_ins_alone, loss_ins_alone = get_perplexity_and_embedding_whole_text(tokenizer, model, question, max_length - instruct_i_reverse_len + 1)
    ppl_ins_condition, loss_ins_condition = get_perplexity_and_embedding_part_text(tokenizer, model, instruct_i_reverse, question, max_length)

    rifd_ppl = ppl_ins_condition / ppl_ins_alone if ppl_ins_alone != 0 else 0
    return rifd_ppl

# Function to calculate IFD (Information Flow Discrepancy)
def calculate_ifd(query):
    output_i = generate_answer(query)
    whole_text = f"{query}[答案]: {output_i}"
    instruct_i_input_ids = tokenizer.encode(query, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
    instruct_i_len = instruct_i_input_ids.shape[1]

    ppl_out_alone, loss_out_alone = get_perplexity_and_embedding_whole_text(tokenizer, model, output_i, max_length - instruct_i_len + 1)
    ppl_out_condition, loss_out_condition = get_perplexity_and_embedding_part_text(tokenizer, model, whole_text, output_i, max_length)

    ifd_ppl = ppl_out_condition / ppl_out_alone if ppl_out_alone != 0 else 0
    return ifd_ppl

# Main script to process data from 'ChatGPT.json' and calculate metrics
if __name__ == "__main__":
    with open('book2qa_sft.json', 'r') as infile:
        group = json.load(infile)

    ifd_sum = 0
    rifd_sum = 0
    count = 0

    for item in group:
        ifd_sum += item['ifd']
        rifd_sum += item['r-ifd']
        count += 1

    average_ifd = ifd_sum / count if count > 0 else 0
    average_rifd = rifd_sum / count if count > 0 else 0

    print(f"Average IFD: {average_ifd}")
    print(f"Average r-IFD: {average_rifd}")
