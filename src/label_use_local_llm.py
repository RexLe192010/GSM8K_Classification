import json
import fire
from tqdm import tqdm
from src.local_llm import LocalLLM
import os
import jsonlines

prompt_template = '''
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: You are given the following question:
{question_text}

You should decide how to solve the question to achieve the best accuracy. Here're two options.
Option A: Directly give the answer. 
Option B: Think it step by step. 
Now answer your selected option (A or B), without adding anything else.
ASSISTANT: Option
'''.strip()

def main():
    local_llm = LocalLLM(os.path.expanduser('~/vicuna-13b'), load_in_8bit=False)

    to_label_question_list = []
    with jsonlines.open('./raw_dataset/mmlu.jsonl') as reader:
        for q in reader:
            to_label_question_list.append(q)
    save_path = '/tmp/id_to_vicuna_direct_or_cot_info.json'

    id_to_vicuna_direct_or_cot_info = {}
    cnt = 0
    for q in tqdm(to_label_question_list):
        prompt = prompt_template.format(question_text=q['text'])
        target_logits = local_llm.logit_inference(prompt, target_tokens=['A', 'B'])
        id_to_vicuna_direct_or_cot_info[q['id']] = {
            'target_logits': [float(v) for v in target_logits],
            'logit_diff': float(target_logits[0] - target_logits[1]),
        }
        cnt += 1
        if cnt % 100 == 0:
            with open(save_path, 'w') as f:
                json.dump(id_to_vicuna_direct_or_cot_info, f)


    with open(save_path, 'w') as f:
        json.dump(id_to_vicuna_direct_or_cot_info, f)

if __name__ == '__main__':
    fire.Fire(main)