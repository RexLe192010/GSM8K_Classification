from src.utils import ChatGPTInternal
import fire
import json
import jsonlines
import os
from tqdm import tqdm
import traceback
import time

prompt_template = '''
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: You are given the following question:
{question_text}

You should decide how to solve the question. Here're two options.
Option A: Directly give the answer. 
Option B: Think it step by step. 
Now answer your selected option (A or B), without adding anything else.
ASSISTANT: Option
'''


def main():
    client = ChatGPTInternal(temp=0.0, limit_rpm=60, max_tokens=5)
    to_label_question_list = []
    with jsonlines.open('./raw_dataset/mmlu.jsonl') as reader:
        for q in reader:
            to_label_question_list.append(q)

    save_path = './raw_dataset/id_to_chatgpt_direct_or_cot_info.json'
    if os.path.exists(save_path):
        with open(save_path) as f:
            id_to_chatgpt_direct_or_cot_info = json.load(f)
    else:
        id_to_chatgpt_direct_or_cot_info = {}
    index = 0
    while True:
        if index >= len(to_label_question_list):
            break
        print("{} / {}".format(index + 1, len(to_label_question_list)))
        q = to_label_question_list[index]
        try:
            if q['id'] in id_to_chatgpt_direct_or_cot_info:
                index += 1
                print('skip {}'.format(q['id']))
                continue
            prompt = prompt_template.format(question_text=q['text'])
            ret, _ = client.call_completion(prompt)
            print(ret)
            if 'error' in ret:
                time.sleep(62)
                continue
            option = ret['choices'][0]['text'][0]
            if option not in ['A', 'B']:
                print("Unexpected response!".format(ret['choices'][0]['text']))
            if option == 'A':
                choice = 'direct'
            elif option == 'B':
                choice = 'cot'
            else:
                choice = 'cot'
            id_to_chatgpt_direct_or_cot_info[q['id']] = choice
            index += 1
        except Exception as e:
            traceback.print_exc()
            index += 1
        if index % 10 == 0:
            with open(save_path, 'w') as f:
                json.dump(id_to_chatgpt_direct_or_cot_info, f)

    with open(save_path, 'w') as f:
        json.dump(id_to_chatgpt_direct_or_cot_info, f)

if __name__ == '__main__':
    fire.Fire(main)