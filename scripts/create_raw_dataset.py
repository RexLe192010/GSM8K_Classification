from datasets import load_dataset
import jsonlines
import hashlib
import fire
import random
import os
import glob
import json

mmlu_question_text_prompt = '''
Question: {input}
Choices:
A. {A}
B. {B}
C. {C}
D. {D}
'''.strip()

ceval_question_text_prompt = '''
问题：{question}
选项：
A. {A}
B. {B}
C. {C}
D. {D}
'''.strip()

def create_gsm8k_dataset():
    full_dataset = load_dataset('gsm8k', 'main')
    train = full_dataset['train']
    test = full_dataset['test']
    dataset = []
    question_list = []
    for sample in train:
        dataset.append(sample)
    for sample in test:
        dataset.append(sample)
    for sample in dataset:
        question_text = sample['question']
        answer_text = sample['answer']
        qid = hashlib.md5("{} {}".format("gsm8k", question_text).encode('utf-8')).hexdigest()
        question = {
            'dataset': 'gsm8k',
            'id': qid,
            'text': question_text,
            'target': answer_text,
        }
        print(question)
        question_list.append(question)
    random.shuffle(question_list)
    with jsonlines.open('../raw_dataset/gsm8k.jsonl', 'w') as writer:
        for question in question_list:
            writer.write(question)


def create_gsm8k_test_dataset():
    full_dataset = load_dataset('gsm8k', 'main')
    dataset = full_dataset['test']
    question_list = []
    for sample in dataset:
        question_text = sample['question']
        answer_text = sample['answer']
        qid = hashlib.md5("{} {}".format("gsm8k", question_text).encode('utf-8')).hexdigest()
        question = {
            'dataset': 'gsm8k',
            'id': qid,
            'text': question_text,
            'target': answer_text,
        }
        print(question)
        question_list.append(question)
    random.shuffle(question_list)
    with jsonlines.open('../raw_dataset/gsm8k_test.jsonl', 'w') as writer:
        for question in question_list:
            writer.write(question)


def create_gsm8k_train_1000_dataset():
    full_dataset = load_dataset('gsm8k', 'main')
    dataset = full_dataset['train']
    question_list = []
    for sample in dataset:
        question_text = sample['question']
        answer_text = sample['answer']
        qid = hashlib.md5("{} {}".format("gsm8k", question_text).encode('utf-8')).hexdigest()
        question = {
            'dataset': 'gsm8k',
            'id': qid,
            'text': question_text,
            'target': answer_text,
        }
        print(question)
        question_list.append(question)
    new_question_list = random.sample(question_list, 1000)
    random.shuffle(new_question_list)
    with jsonlines.open('../raw_dataset/gsm8k_train_1000.jsonl', 'w') as writer:
        for question in new_question_list:
            writer.write(question)



def create_ceval_dataset():
    ceval_subjects = ['computer_network', 'operating_system', 'computer_architecture', 'college_programming', 'college_physics', 'college_chemistry', 'advanced_mathematics', 'probability_and_statistics', 'discrete_mathematics', 'electrical_engineer', 'metrology_engineer', 'high_school_mathematics', 'high_school_physics', 'high_school_chemistry', 'high_school_biology', 'middle_school_mathematics', 'middle_school_biology', 'middle_school_physics', 'middle_school_chemistry', 'veterinary_medicine', 'college_economics', 'business_administration', 'marxism', 'mao_zedong_thought', 'education_science', 'teacher_qualification', 'high_school_politics', 'high_school_geography', 'middle_school_politics', 'middle_school_geography', 'modern_chinese_history', 'ideological_and_moral_cultivation', 'logic', 'law', 'chinese_language_and_literature', 'art_studies', 'professional_tour_guide', 'legal_professional', 'high_school_chinese', 'high_school_history', 'middle_school_history', 'civil_servant', 'sports_science', 'plant_protection', 'basic_medicine', 'clinical_medicine', 'urban_and_rural_planner', 'accountant', 'fire_engineer', 'environmental_impact_assessment_engineer', 'tax_accountant', 'physician']
    question_list = []
    for subject in ceval_subjects:
        full_dataset = load_dataset("ceval/ceval-exam", subject)    
        for origin_split in ['val']:
            dataset = full_dataset[origin_split]
            for i in range(len(dataset)):
                origin_q = dataset[i]
                question_text = ceval_question_text_prompt.format(**origin_q)
                qid = hashlib.md5("{} {}".format("ceval", question_text).encode('utf-8')).hexdigest()
                question = {
                    'dataset': 'ceval',
                    'id': qid,
                    'text': question_text,
                    'choice_num': 4,
                    'target': origin_q['answer'],
                    'metadata': {
                        'origin_split': origin_split,
                        'subject': subject,
                    }
                }
                question_list.append(question)
    random.shuffle(question_list)
    with jsonlines.open('./raw_dataset/ceval.jsonl', 'w') as writer:
        for question in question_list:
            writer.write(question)

def create_mmlu_dataset():
    mmlu_subjects =  [
        'high_school_european_history', 'business_ethics', 
        'clinical_knowledge', 'medical_genetics', 'high_school_us_history', 'high_school_physics', 
        'high_school_world_history', 'virology', 'high_school_microeconomics', 'econometrics', 'college_computer_science', 
        'high_school_biology', 'abstract_algebra', 'professional_accounting', 'philosophy', 'professional_medicine', 
        'nutrition', 'global_facts', 'machine_learning', 'security_studies', 'public_relations', 'professional_psychology',
        'prehistory', 'anatomy', 'human_sexuality', 'college_medicine', 'high_school_government_and_politics', 
        'college_chemistry', 'logical_fallacies', 'high_school_geography', 'elementary_mathematics', 'human_aging', 
        'college_mathematics', 'high_school_psychology', 'formal_logic', 'high_school_statistics', 'international_law', 
        'high_school_mathematics', 'high_school_computer_science', 'conceptual_physics', 'miscellaneous', 
        'high_school_chemistry', 'marketing', 'professional_law', 'management', 'college_physics', 'jurisprudence', 
        'world_religions', 'sociology', 'us_foreign_policy', 'high_school_macroeconomics', 'computer_security', 
        'moral_scenarios', 'moral_disputes', 'electrical_engineering', 'astronomy', 'college_biology'
    ]
    question_list = []
    idx = 0
    for subject in mmlu_subjects:
        origin_split = 'test'
        dataset = load_dataset("lukaemon/mmlu", subject)[origin_split]
        for i in range(len(dataset)):
            origin_q = dataset[i]
            question_text = mmlu_question_text_prompt.format(**origin_q)
            qid = hashlib.md5("{} {}".format("mmlu", question_text).encode('utf-8')).hexdigest()
            question = {
                'dataset': 'mmlu',
                'id': qid,
                'text': question_text,
                'choice_num': 4,
                'target': origin_q['target'],
                'metadata': {
                    'origin_split': origin_split,
                    'subject': subject,
                }
            }
            question_list.append(question)
    random.shuffle(question_list)
    with jsonlines.open('./raw_dataset/mmlu.jsonl', 'w') as writer:
        for question in question_list:
            writer.write(question)

def create_bbh_dataset():
    bbh_repo_path = '~/BIG-Bench-Hard/'
    question_list = []
    for path in glob.glob(os.path.join(os.path.expanduser(bbh_repo_path), 'bbh', "*.json")):
        with open(path) as f:
            j = json.load(f)
            bbh_task_name = os.path.basename(path).split('.')[0]
            for example in j['examples']:
                question_text = example['input']
                target = example['target']
                split = "TEST"
                id = hashlib.md5("{} {}".format("bbh", question_text).encode('utf-8')).hexdigest()
                question_list.append({
                    'dataset': 'bbh',
                    'id': id,
                    'text': question_text,
                    'target': target,
                    'metadata': {
                        'task_name': bbh_task_name,
                    }
                })
    random.shuffle(question_list)
    with jsonlines.open('./raw_dataset/bbh.jsonl', 'w') as writer:
        for question in question_list:
            writer.write(question)

if __name__ == '__main__':
    fire.Fire(create_gsm8k_train_1000_dataset)