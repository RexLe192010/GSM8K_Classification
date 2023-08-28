from src.utils import ChatGPTInternal
import fire
import jsonlines
import traceback
import time

prompt_template = '''
You are given the following question:

{question_text}

Output your answer directly (A/B/C/D) without adding anything else.
'''.strip()


def main():
    client = ChatGPTInternal(temp=0.0, limit_rpm=60, max_tokens=5)
    to_label_question_list = []
    with jsonlines.open('./raw_dataset/mmlu.jsonl') as reader:
        for q in reader:
            to_label_question_list.append(q)
    index = 0
    correct = 0
    while True:
        if index >= len(to_label_question_list):
            break
        print("{} / {}".format(index + 1, len(to_label_question_list)))
        q = to_label_question_list[index]
        try:
            prompt = prompt_template.format(question_text=q['text'])
            ret, _ = client.call_chat(prompt)
            print(ret)
            if 'error' in ret:
                time.sleep(62)
                continue
            choice = ret['choices'][0]['message']['content'].strip().strip('.').strip()[0]
            if choice == q['target']:
                correct += 1
            index += 1
            print("Acc: {:.2f}%".format(correct / index * 100))
        except Exception as e:
            traceback.print_exc()
            index += 1


if __name__ == '__main__':
    fire.Fire(main)