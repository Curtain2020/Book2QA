import json
import jieba
from collections import Counter
import os

def calculate_k_precision(standard_answer, response):
    """
    使用 jieba 分词计算给定标准答案和模型响应的 k-Precision。
    """
    # 使用 jieba 进行分词
    standard_tokens = set(jieba.cut(standard_answer))
    response_tokens = list(jieba.cut(response))
    
    # 计算共同词汇的数量
    common_tokens = Counter(standard_tokens) & Counter(response_tokens)
    num_common = sum(common_tokens.values())
    
    # 计算 k-Precision
    if len(response_tokens) == 0:
        return 0  # 避免除以零
    return num_common / len(response_tokens)

def update_json_file(file_path):
    """
    读取 JSON 文件，计算 k-Precision，并将结果写回文件。
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # 为每个条目计算 k-Precision 并更新数据
    for entry in data:
        kprecision = calculate_k_precision(entry['standard_answer'], entry['response'])
        entry['kprecision'] = kprecision

    # 将更新后的数据写回原文件
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def process_directory(directory_path):
    """
    处理指定目录下的所有 JSON 文件。
    """
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            update_json_file(file_path)
            print(f"Processed {filename}")

# 指定目录路径
directory_path = '/public/home/wxhu/llm/zhcui/book2qa/book2qa_metric_eval/CMRC_2018_result'
process_directory(directory_path)
