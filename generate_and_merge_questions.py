import json
import torch
from sklearn.metrics.pairwise import euclidean_distances
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
from collections import defaultdict
from tqdm import tqdm
import random

# 定义模型列表
model_names = [
    "Qwen/Qwen-7B-Chat", 
    "internlm/internlm-chat-7b", 
    "baichuan-inc/Baichuan2-13B-Chat"
]

def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.eval()
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    return model, tokenizer

def generate_answer(model, tokenizer, instruction):
    messages = []
    messages.append({"role": "user", "content": instruction})
    generated_text = model.chat(tokenizer, messages)
    return generated_text

def generate_instruction(paragraph, prompt, key_word, in_related_summary, out_related_summary, flag):
    if flag == 0:
        instruction = ('##指令\n你是一个出题人，基于[文档]，请提出一个[问题]，提出的[问题]要以[文档]的要点为基础，并满足以下要求：\n{}\n'
                       '[问题]要尽可能复杂。\n'
                       '##文档\n{}\n'
                       '##输出格式\n\n[问题]：<提出的一个问题>\n\n'
                       '##注意：只需要给出一个[问题]，不要添加其他额外信息或解释。\n'
                       ).format(prompt, paragraph)
    elif flag == 1:
        instruction = ('##指令\n你是一个出题人，基于[文档]，请提出一个[问题]，提出的[问题]要以[文档]的要点为基础，并满足以下要求：\n{}\n'
                       '提出的[问题]要关于[主题]:**{}**\n'
                       '[问题]要尽可能复杂。\n'
                       '##文档\n{}\n'
                       '##输出格式\n\n[问题]：<提出的一个问题>\n\n'
                       '##注意：只需要给出一个[问题]，不要添加其他额外信息或解释。\n'
                       ).format(prompt, key_word, paragraph)
    elif flag == 2:
        instruction = ('##指令\n你是一个出题人，基于[文档]，请提出一个[问题]，提出的[问题]要以[文档]的要点为基础，当[问题]需要额外信息时再参考[参考摘要],并满足以下要求：\n{}\n'
                       '[问题]要尽可能复杂。\n'
                       '##参考摘要\n与[文档]主题相同的[参考摘要]:{}\n与[文档]主题不同的[参考摘要]:{}\n'
                       '##文档\n{}\n'
                       '##输出格式\n\n[问题]：<提出的一个问题>\n\n'
                       '##注意：只需要给出一个[问题]，不要添加其他额外信息或解释。\n'
                       ).format(prompt, in_related_summary, out_related_summary, paragraph)
    return instruction

def process_json(label_now):
    with open("../dataset/clusters.json", "r", encoding="utf-8") as file:
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

def response_question(input_file, output_file, model, tokenizer):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for section_key, section_value in data.items():
        for section in section_value['sections']:
            for subsection in section['subsections']:
                paragraph = subsection['paragraphs']
                summarization = subsection['summarization']
                label_now = subsection['cluster_id']
                key_words = subsection['key_words']
                results_list = []
                in_summary, out_summary = process_json(label_now)
                prompt_list = [
                    '提出认知性问题，对[文档]中知识的回忆与确认。用一种非常接近于学生当初遇到的某种观念和现象时的形式，进行提问。提示词：回忆，记忆，识别，列表，定义，陈述，呈现',
                    '提出理解性问题，对[文档]中有关概念内容的理解进行提问，[问题]的回答需要根据[文档]进行总结和比较。提示词: 说明，识别，描述，解释，区别，重述，归纳，比较',
                    '提出分析性问题，问题的回答要利用[文档]中的内容进行推理与解析，详细地阐明基础理论和基本原理。提示词: 分析，检查，实验，组织，对比，比较，辨别，区别',
                    '提出评价性问题，问题的回答要根据[文档]作出合适的判断，并提出有说服力的观点。提示词: 评价，估计，评论，鉴定，辩明，辩护，证明，预测，预言，支持',
                    '提出应用性问题，问题的回答要利用[文档]和[参考摘要]中的信息解决实际的问题，涉及具有特色的表达，制定合理的计划和可实施的步骤，根据基本材料推出某种规律等活动。提示词: 应用，论证，操作，实践，分类，举例说明，解决',
                    '提出综合性问题，对[文档]的内容进行发散，可以做一些假设，利用[文档]和[参考摘要]的内容进行系统的分析，问题的回答要对事物本质的价值作出有说服力的判断。提示词: 组成，建立，设计，开发，计划，支持，系统化'
                ]
                for prompt in prompt_list:
                    instruction = generate_instruction(paragraph, prompt, '', '', '', 0)
                    output = generate_answer(model, tokenizer, instruction)
                    results_list.append(output)
                for key_word in key_words:
                    for prompt in prompt_list:
                        instruction = generate_instruction(paragraph, prompt, key_word, '', '', 1)
                        output = generate_answer(model, tokenizer, instruction)
                        results_list.append(output)
                for prompt in prompt_list:
                    instruction = generate_instruction(paragraph, prompt, '', in_summary, out_summary, 2)
                    output = generate_answer(model, tokenizer, instruction)
                    results_list.append(output)
                subsection['questions'] = results_list

    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(data, f_out, ensure_ascii=False, indent=4)

def merge_questions(output_file):
    file_paths = ['qwen-7b-chat.json', 'internlm-7b-chat.json', 'baichuan-13b-chat.json']
    file_names = ['qwen-7b-chat', 'internlm-7b-chat', 'baichuan-13b-chat']

    data_list = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data = json.load(file)
            data_list.append(data)

    merged_data = defaultdict(list)

    for data, file_name in zip(data_list, file_names):
        for key, value in data.items():
            if key in ['数据与编程正文_三样', '信息系统正文_三样']:
                for section in value['sections']:
                    for subsection in section['subsections']:
                        name = subsection['name']
                        questions = subsection['questions']
                        new_questions = [{"question": question, "ifd": None} for question in questions if len(question) <= 400 and len(question) >= 5]
                        merged_data[name].append({"file_name": file_name, "questions": new_questions})

    output_data = {
        "merged_data": [{"name": name, "group": group} for name, group in merged_data.items()]
    }

    with open(output_file, 'w') as outfile:
        json.dump(output_data, outfile, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    for model_name in model_names:
        model, tokenizer = load_model_and_tokenizer(model_name)
        output_file = f"{model_name.split('/')[-1]}-chat.json"
        response_question('../dataset/book_cluster.json', output_file, model, tokenizer)
    merge_questions('../dataset/merged_question.json')
