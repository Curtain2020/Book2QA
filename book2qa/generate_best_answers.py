import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from tqdm import tqdm
import random

# Define model list
model_names = [
    "Qwen/Qwen-7B-Chat",
    "internlm/internlm-chat-7b",
    "baichuan-inc/Baichuan2-13B-Chat"
]

# Function to load model and tokenizer
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.eval()
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    return model, tokenizer

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

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
def generate_answer(model, tokenizer, instruction):
    messages = [{"role": "user", "content": instruction}]
    generated_text = model.chat(tokenizer, messages)
    if "[答案]:" in generated_text:
        answer_index = generated_text.index("[答案]:") + len("[答案]:")
        answer = generated_text[answer_index:].strip()
    else:
        answer = generated_text
    return answer

# Function to generate an instruction
def generate_instruction(paragraph, question, in_related_summary, out_related_summary):
    instruction = ('##指令\n基于[文档]，回答[问题]，[答案]要以[文档]的要点为基础，当[答案]需要额外信息时再参考[参考摘要]\n'
                   '##问题\n{}\n'
                   '##文档\n{}\n'
                   '##参考摘要\n与[文档]主题相同的[参考摘要]:{}\n与[文档]主题不同的[参考摘要]:{}\n'
                   '##输出格式\n\n[答案]：\n\n'
                   ).format(question, paragraph, in_related_summary, out_related_summary)
    return instruction

# Function to process JSON and get summaries
def process_json(label_now):
    with open("clusters.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    in_related_sentences = []
    out_related_sentences = []

    def get_random_sentences(sentences, count):
        return random.sample(sentences, count)

    for item in data:
        label = item["label"]
        sentences = item["sentence"]

        if label == label_now:
            in_related_sentences.extend(get_random_sentences(sentences, 7))
        else:
            out_related_sentences.append(random.choice(sentences))

    in_related_summary = "\n".join(in_related_sentences)
    out_related_summary = "\n".join(out_related_sentences)

    return in_related_summary, out_related_summary

# Function to calculate r-ifd
def calculate_r_ifd(tokenizer, model, question, answer):
    max_length = 2048
    instruct_i_reverse = '以下是对一个[问题]的[回答]，请猜测给定[回答]的相应[问题]。[回答]：' + answer + '\n生成上述[回答]的[问题]。[问题]：'
    whole_text_reverse = instruct_i_reverse + question

    instruct_i_reverse_input_ids = tokenizer.encode(instruct_i_reverse, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    instruct_i_reverse_len = instruct_i_reverse_input_ids.shape[1]

    ppl_ins_alone, loss_ins_alone = get_perplexity_and_embedding_whole_text(tokenizer, model, question, max_length - instruct_i_reverse_len + 1)
    ppl_ins_condition, loss_ins_condition = get_perplexity_and_embedding_part_text(tokenizer, model, whole_text_reverse, question, max_length)

    rifd_ppl = ppl_ins_condition / ppl_ins_alone if ppl_ins_alone != 0 else 0
    return rifd_ppl

# Function to process the JSON file and generate answers
def response_question(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for subsection in data['merged_data']:
        paragraph = subsection['paragraphs']
        summarization = subsection['summarization']
        label_now = subsection['cluster_id']
        in_summary, out_summary = process_json(label_now)

        for item in subsection['group']:
            instruction = generate_instruction(paragraph, item['question'], in_summary, out_summary)

            # Generate answers from different models
            model_answers = []
            for model_name in model_names:
                model, tokenizer = load_model_and_tokenizer(model_name)
                answer = generate_answer(model, tokenizer, instruction)
                r_ifd = calculate_r_ifd(tokenizer, model, item['question'], answer)
                model_answers.append({"answer": answer, "r_ifd": r_ifd})

            # Select the answer with the smallest r-ifd
            best_answer = min(model_answers, key=lambda x: x["r_ifd"])
            item['answer'] = best_answer["answer"]
            item['r-ifd'] = best_answer["r_ifd"]

    # Write the final result to output file
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(data, f_out, ensure_ascii=False, indent=4)

response_question('../dataset/generated_questions.json', '../dataset/generated_answers.json')

# Function to create SFT data
def create_sft_data(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    formatted_data = []

    for merged_item in data['merged_data']:
        for group_item in merged_item['group']:
            instruction = ""
            input_data = group_item['question']
            output_data = group_item['answer']
            
            formatted_item = {
                'instruction': instruction,
                'input': input_data,
                'output': output_data
            }
            formatted_data.append(formatted_item)

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(formatted_data, file, ensure_ascii=False, indent=4)

create_sft_data('../dataset/generated_answers.json', '../dataset/book2qa_sft.json')
