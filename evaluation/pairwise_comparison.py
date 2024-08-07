from openai import OpenAI
import httpx
import json

# Initialize OpenAI client with base URL and API key
client = OpenAI(
    api_key="your-api-key-here",  # Replace with your actual API key for GPT-4
    http_client=httpx.Client(
        base_url="https://api.openai.com/v1",  # OpenAI 官方 API 的 URL
        follow_redirects=True,
    ),
)

def generate_instruction(question, assistant1, assistant2):
    # Generate the instruction prompt based on provided question and answers from assistants
    instruction = ('#instruction\n'
                   '你是一位负责检查答案质量的助手，需提供有帮助且精确的反馈。\n\n'
                   '#问题#\n'
                   '{}\n\n'
                   '#答案#\n'
                   '[助手2答案的开始]\n'
                   '{}\n'
                   '[助手2答案的结束]\n'
                   '[助手1答案的开始]\n'
                   '{}\n'
                   '[助手1答案的结束]\n\n'
                   '我们希望你对上面显示问题的两个AI助手的表现提供反馈。你的评价应考虑到诸如帮助性、相关性、准确性、深度、创造力和详细程度等因素。每个助手的总体得分为1到5分，分数越高表示整体表现越好。请首先输出一行，仅包含助手1和助手2的分数，分别以空格分隔。在随后的行中，请提供评估的综合解，避免任何潜在的偏见，并确保回答的顺序不会影响判断。\n').format(question, assistant2, assistant1)
    return instruction

def call_with_messages(instruction):
    # Call the GPT-4 API to generate completions based on the provided instruction
    completion = client.chat.completions.create(
        model="gpt-4",  # Specify GPT-4 model name or ID
        temperature=0,
        top_p=0.5,
        messages=[
            {"role": "user", "content": instruction}
        ]
    )
    return completion

def process_json(input_file):
    # Process JSON input file to read question, assistant1, and assistant2 for each pair of data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Assuming 'data' is a list of dictionaries, each containing 'question', 'assistant1', 'assistant2'
    for item in data:
        question = item.get('question', '')
        assistant1 = item.get('assistant1', '')
        assistant2 = item.get('assistant2', '')

        instruction = generate_instruction(question, assistant1, assistant2)
        print(instruction)
        
        # Call GPT-4 API to generate completions for each item
        result = call_with_messages(instruction)
        print(result.choices[0].message.content)  # Output generated by GPT-4
        
        # Uncomment to write results to a JSON file for each item
        # with open('eval2.json', 'w', encoding='utf-8') as f:
        #     json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    input_file = 'pair_result.json'  # Update to your actual input file name
    process_json(input_file)
