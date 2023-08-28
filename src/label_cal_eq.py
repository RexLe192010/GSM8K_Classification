import json
import fire
import time
from tqdm import tqdm
from src.local_llm import LocalLLM
import os
import jsonlines
from src.chat_api import ChatGPTInternal



vicuna_prompt_template = '''
# Task
Your task is to decide how to solve the question that achieves the best accuracy.


# Options
You should choose between A and B.
A: Do not set up equations.
B: Set up equations. 


# Examples

Question:
In 5 years, Joey will be as old as Beth is now. If Joey is 9 now, how old was Joey when Beth was Joey's age now?
Thinking:
This problem is better suited for solving using equations because it involves a specific mathematical relation between Joey and Beth's ages and requires the establishment of an equation to find a solution.
Equations provide a more formal and general way to represent such abstract relationships.
#### B

Question:
Inez has $150. She spends one-half on hockey skates and a certain amount on hockey pads. If Inez has $25 remaining, how much did the hockey pads cost, together, in dollars?
Thinking:
This problem is better suited for arithmetic rather than equations because it involves relatively straightforward relationships that can be solved through direct calculations.
The problem presents relatively simple relationships, such as Inez spending half of the total amount on hockey skates and the remaining amount on hockey pads.
In such cases, we can solve the problem through straightforward arithmetic operations without introducing algebraic equations.
#### A

Question:
Jean is two years older than Mark. Two years ago Mark was 5 years older than half Jan's age. If Jan is 30 how old is Jean?
Thinking:
The problem introduces multiple unknown age variables, namely the ages of Jean and Mark.
#### B

Question:
Danivan Drugstore has 4500 bottles of hand sanitizer gel in inventory at the beginning of the week. On Monday 2445 bottles were sold, on Tuesday 900 bottles were sold, and 50 bottles each day for the rest of the week were sold (from Wednesday until Sunday). On Saturday, the supplier delivers an order for 650 bottles. How many bottles of sanitizer gel does the Drugstore have at the end of the week?
Thinking:
This problem is better suited for solving using direct arithmetic calculations rather than equations because it provides specific quantities and operations that can be straightforwardly computed step by step.
#### A

Question:
Nina has four times more math homework and eight times more reading homework than Ruby. If Ruby has six math homework and two reading homework, how much homework is Nina having altogether?
Thinking:
This problem is better suited for solving using direct arithmetic calculations because the relationships involved are relatively straightforward and can be solved using basic arithmetic operations without the need for complex equations.
#### A

Question:
In today's field day challenge, the 4th graders were competing against the 5th graders.  Each grade had 2 different classes.  The first 4th grade class had 12 girls and 13 boys.  The second 4th grade class had 15 girls and 11 boys.  The first 5th grade class had 9 girls and 13 boys while the second 5th grade class had 10 girls and 11 boys.  In total, how many more boys were competing than girls?
Thinking:
This problem is better suited for solving using equations because it involves a comparison of student numbers between two different grades and requires finding the difference in quantity between boys and girls.
#### B


# Output

You are given the following question,

Question:
{question_text}

Output your thinking and your choice.
You should include your choice after #### (Don't add anything else after A or B)
'''.strip()

chat_prompt_template = '''
# Task
Your task is to decide how to solve the question that achieves the best accuracy.


# Options
You should choose between A and B.
A: Do not set up equations.
B: Set up equations. 


# Examples

Question:
In 5 years, Joey will be as old as Beth is now. If Joey is 9 now, how old was Joey when Beth was Joey's age now?
Thinking:
This problem is better suited for solving using equations because it involves a specific mathematical relation between Joey and Beth's ages and requires the establishment of an equation to find a solution.
Equations provide a more formal and general way to represent such abstract relationships.
#### B

Question:
Inez has $150. She spends one-half on hockey skates and a certain amount on hockey pads. If Inez has $25 remaining, how much did the hockey pads cost, together, in dollars?
Thinking:
This problem is better suited for arithmetic rather than equations because it involves relatively straightforward relationships that can be solved through direct calculations.
The problem presents relatively simple relationships, such as Inez spending half of the total amount on hockey skates and the remaining amount on hockey pads.
In such cases, we can solve the problem through straightforward arithmetic operations without introducing algebraic equations.
#### A

Question:
Jean is two years older than Mark. Two years ago Mark was 5 years older than half Jan's age. If Jan is 30 how old is Jean?
Thinking:
The problem introduces multiple unknown age variables, namely the ages of Jean and Mark.
#### B

Question:
Danivan Drugstore has 4500 bottles of hand sanitizer gel in inventory at the beginning of the week. On Monday 2445 bottles were sold, on Tuesday 900 bottles were sold, and 50 bottles each day for the rest of the week were sold (from Wednesday until Sunday). On Saturday, the supplier delivers an order for 650 bottles. How many bottles of sanitizer gel does the Drugstore have at the end of the week?
Thinking:
This problem is better suited for solving using direct arithmetic calculations rather than equations because it provides specific quantities and operations that can be straightforwardly computed step by step.
#### A

Question:
Nina has four times more math homework and eight times more reading homework than Ruby. If Ruby has six math homework and two reading homework, how much homework is Nina having altogether?
Thinking:
This problem is better suited for solving using direct arithmetic calculations because the relationships involved are relatively straightforward and can be solved using basic arithmetic operations without the need for complex equations.
#### A

Question:
In today's field day challenge, the 4th graders were competing against the 5th graders.  Each grade had 2 different classes.  The first 4th grade class had 12 girls and 13 boys.  The second 4th grade class had 15 girls and 11 boys.  The first 5th grade class had 9 girls and 13 boys while the second 5th grade class had 10 girls and 11 boys.  In total, how many more boys were competing than girls?
Thinking:
This problem is better suited for solving using equations because it involves a comparison of student numbers between two different grades and requires finding the difference in quantity between boys and girls.
#### B


# Output

You are given the following question,

Question:
{question_text}

Output your thinking and your choice.
You should include your choice after #### (Don't add anything else after A or B)
'''.strip()


completion_prompt_template = '''
There're two options to choose from in order to achieve the best accuracy:
Option A: Do not set up equations
Option B: Set up equations. 

Question:
In 5 years, Joey will be as old as Beth is now. If Joey is 9 now, how old was Joey when Beth was Joey's age now?
Answer:
B

Question:
Jean is two years older than Mark. Two years ago Mark was 5 years older than half Jan's age. If Jan is 30 how old is Jean?
Answer:
B

Question:
{question_text}
Answer:
'''.strip()


def label_by_vicuna_llm():
    local_llm = LocalLLM(os.path.expanduser('~/vicuna-13b'), load_in_8bit=False)

    to_label_question_list = []
    # with jsonlines.open('./raw_dataset/gsm8k.jsonl') as reader:
    #     for q in reader:
    #         to_label_question_list.append(q)
    with jsonlines.open('./raw_dataset/gsm8k_test.jsonl') as reader:
        for q in reader:
            to_label_question_list.append(q)
    save_path = '../GSM8K-classification/20230711/dataset/vicuna_cal_or_eq_test.json'
    # This file will be moved to GSM8K classification directory for further experiments.

    vicuna_cal_or_eq_info = {}
    if os.path.exists(save_path):
        with open(save_path, 'r') as reader:
            vicuna_cal_or_eq_info = json.load(reader)
    cnt = 0
    eq_count = 0
    for q in tqdm(to_label_question_list):
        prompt = vicuna_prompt_template.format(question_text=q['text'])
        target_logits = local_llm.logit_inference(prompt, target_tokens=['A', 'B'])
        logits_list = [float(v) for v in target_logits]
        logit_diff = float(target_logits[0] - target_logits[1])
        if logit_diff < 0:
            # print(prompt)
            # print("Equation!\n")
            eq_count += 1
        vicuna_cal_or_eq_info[q['id']] = {
            'target_logits': logits_list,
            'logit_diff': logit_diff,
        }
        cnt += 1
        if cnt % 100 == 0:
            with open(save_path, 'w') as f:
                json.dump(vicuna_cal_or_eq_info, f)

    print("There are {} equation-suitable questions.".format(eq_count))

    with open(save_path, 'w') as f:
        json.dump(vicuna_cal_or_eq_info, f)


def label_by_chat_llm():
    chat_llm = ChatGPTInternal()

    to_label_question_list = []
    # with jsonlines.open('./raw_dataset/gsm8k.jsonl') as reader:
    #     for q in reader:
    #         to_label_question_list.append(q)
    with jsonlines.open('./raw_dataset/gsm8k_test.jsonl') as reader:
        for q in reader:
            to_label_question_list.append(q)
    print(len(to_label_question_list))
    # save_path = '../GSM8K-classification/20230711/dataset/chat_cal_or_eq.json'
    save_path = '../GSM8K-classification/20230711/dataset/chat_cal_or_eq_test.json'
    # This file will be moved to GSM8K classification directory for further experiments.


    chat_cal_or_eq_info = {}
    if os.path.exists(save_path):
        with open(save_path, 'r') as reader:
            chat_cal_or_eq_info = json.load(reader)
    cnt = 0
    eq_count = 0
    for q in tqdm(to_label_question_list):
        if q['id'] in chat_cal_or_eq_info:
            print("This problem has been classified, skip!")
            continue
        prompt = chat_prompt_template.format(question_text=q['text'])
        response = chat_llm.call_chat(prompt)
        # print(response) # The response is a tuple: (['Option A'], time_cost)
        # print(type(response)) # The type of the response is tuple
        answer_text = response[0][0]
        print(answer_text)
        result = answer_text.split('####')[-1].strip()
        # print(result)
        time_cost = response[1]
        if result == 'B':
            # print(prompt)
            # print("Equation!\n")
            eq_count += 1
        chat_cal_or_eq_info[q['id']] = {
            'raw_response': answer_text,
            'result': result,
            'time_cost': time_cost
        }
        cnt += 1
        if cnt % 20 == 0:
            with open(save_path, 'w') as f:
                json.dump(chat_cal_or_eq_info, f)

    print("There are {} equation-suitable questions in the test dataset.".format(eq_count))
    # 25 equation-suitable questions out of 1319 questions in the test dataset

    with open(save_path, 'w') as f:
        json.dump(chat_cal_or_eq_info, f)


def label_by_completion_llm(): 
    completion_llm = ChatGPTInternal()

    to_label_question_list = []
    with jsonlines.open('./raw_dataset/gsm8k.jsonl') as reader:
        for q in reader:
            to_label_question_list.append(q)
    save_path = '../GSM8K-classification/20230711/dataset/completion_cal_or_eq.json'
    # This file will be moved to GSM8K classification directory for further experiments.

    completion_cal_or_eq_info = {}
    cnt = 0
    eq_count = 0
    for q in tqdm(to_label_question_list):
        prompt = completion_prompt_template.format(question_text=q['text'])
        response = completion_llm.call_completion(prompt)
        print(response)
        print(type(response))
        if response == 'B':
            # print(prompt)
            # print("Equation!\n")
            eq_count += 1
        completion_cal_or_eq_info[q['id']] = {
            'response': response
        }
        cnt += 1
        if cnt % 100 == 0:
            with open(save_path, 'w') as f:
                json.dump(completion_cal_or_eq_info, f)

    print("There are {} equation-suitable questions.".format(eq_count))

    with open(save_path, 'w') as f:
        json.dump(completion_cal_or_eq_info, f)


if __name__ == '__main__':
    fire.Fire(label_by_vicuna_llm)
    # fire.Fire(label_by_chat_llm)
    # fire.Fire(label_by_completion_llm)