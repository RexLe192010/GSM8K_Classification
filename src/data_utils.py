import jsonlines
import numpy as np
from collections import defaultdict
import pandas as pd

def get_labeled_question_list(label_jsonl_path):
    raw_all_cnt = 0
    labeled_question_list = []
    drop_due_to_large_time_cost_cnt = 0
    drop_due_to_none_entry_cnt = 0
    with jsonlines.open(label_jsonl_path) as reader:
        for question in reader:
            raw_all_cnt += 1
            if question['solver_info']['time_cost'] > 60:
                drop_due_to_large_time_cost_cnt += 1
                continue
            if None in [question['solver_info']['is_true'], question['solver_info']['time_cost'], question['solver_info']['dollar_cost']]:
                drop_due_to_none_entry_cnt += 1
                continue
            labeled_question_list.append(question)
    print("Dropped due to large time cost Count: {} Percent: {:.4f}%".format(
        drop_due_to_large_time_cost_cnt,
        drop_due_to_large_time_cost_cnt / raw_all_cnt * 100,
    ))
    print("Dropped due to none entry Count: {} Percent: {:.4f}%".format(
        drop_due_to_none_entry_cnt,
        drop_due_to_none_entry_cnt / raw_all_cnt * 100,
    ))
    return labeled_question_list


def update_solver(question, sample, solver_name, average_metrics):
    rng = np.random.default_rng(1)
    # defaulting
    sample["{}_accuracy".format(solver_name)] = None
    sample["{}_time_cost".format(solver_name)] = None
    sample["{}_dollar_cost".format(solver_name)] = None
    # start
    metrics_name_list = ['is_true', 'time_cost', 'dollar_cost']
    metrics = defaultdict(list)
    cnt = 0
    for solver_info in question['solver_info_list']:
        if solver_info['name'] == solver_name:
            for metrics_name in metrics_name_list:
                metrics[metrics_name].append(solver_info[metrics_name])
            cnt += 1
    for metrics_name in metrics_name_list:
        one_metrics_list = metrics[metrics_name]
        if len(one_metrics_list) == 0:
            sample['{}_{}'.format(solver_name, metrics_name)] = None
        else:
            if average_metrics is False:
                # for fairness against majority vote, use a random selection here.
                sample['{}_{}'.format(solver_name, metrics_name)] = float(rng.choice(one_metrics_list))
            else:
                sample['{}_{}'.format(solver_name, metrics_name)] = float(np.mean(one_metrics_list))
    sample['{}_accuracy'.format(solver_name)] = sample['{}_is_true'.format(solver_name)]
    del sample['{}_is_true'.format(solver_name)]
    return sample

def update_solver_majority_vote(question, sample, solver_name):
    rng = np.random.default_rng(10)
    assert '_maj@' in solver_name
    origin_solver_name = solver_name.split('_maj@')[0]
    maj_k = int(solver_name.split('_maj@')[1])
    # defaulting
    sample["{}_accuracy".format(solver_name)] = None
    sample["{}_time_cost".format(solver_name)] = None
    sample["{}_dollar_cost".format(solver_name)] = None
    # start
    metrics_name_list = ['is_true', 'time_cost', 'dollar_cost']
    metrics = defaultdict(list)
    anwser_list = []
    cnt = 0
    for solver_info in question['solver_info_list']:
        if solver_info['name'] == origin_solver_name:
            for metrics_name in metrics_name_list:
                metrics[metrics_name].append(solver_info[metrics_name])
            anwser_list.append(solver_info['ans'])
            cnt += 1
    if cnt >= maj_k:
        # must accomplish for k times
        # different metrics have different ways to combine
        selected_indices = rng.choice(list(range(cnt)), maj_k, replace=False)
        time_cost_list = metrics['time_cost']
        dollar_cost_list = metrics['dollar_cost']
        # maximum time cost
        time_cost = max([time_cost_list[idx] for idx in selected_indices])
        sample["{}_time_cost".format(solver_name)] = time_cost
        # sum of dollar cost
        dollar_cost = sum([dollar_cost_list[idx] for idx in selected_indices])
        sample["{}_dollar_cost".format(solver_name)] = dollar_cost
        # majority_vote
        answer_count = defaultdict(int)
        selected_ans_list = [anwser_list[idx] for idx in selected_indices]
        for ans in selected_ans_list:
            answer_count[ans] += 1
        maj_count = max(answer_count.values())
        maj_ans_list = []
        for ans, count in answer_count.items():
            if count == maj_count:
                maj_ans_list.append(ans)
        maj_selected_ans = rng.choice(maj_ans_list)
        if maj_selected_ans == question['target']:
            sample["{}_accuracy".format(solver_name)] = 1.0
        else:
            sample["{}_accuracy".format(solver_name)] = 0.0
    return sample

def get_experience_question_list(label_jsonl_path_list, valid_solver_name_list):
    all_labeled_question_list = []
    qid_to_question = {}
    for label_jsonl_path in label_jsonl_path_list:
        labeled_question_list = get_labeled_question_list(label_jsonl_path)
        for question in labeled_question_list:
            # filter_abnormal_data
            qid_to_question[question['id']] = {
                'id': question['id'],
                'dataset': question['dataset'],
                'text': question['text'],
                'target': question['target'],
                'choice_num': question['choice_num'] if 'choice_num' in question else None,
                'metadata': question['metadata'],
                'solver_info_list': [],
            }
        all_labeled_question_list.extend(labeled_question_list)
    for question in all_labeled_question_list:
        qid_to_question[question['id']]['solver_info_list'].append(question['solver_info'])
    experience_question_list = [q for _, q in qid_to_question.items()]
    return experience_question_list

def create_sample_df(
    experience_question_list, 
    normal_solver_name_list, 
    train_ratio = 0.7, dev_ratio = 0.1, test_ratio = 0.2, 
    maj_solver_name_list=[],
    average_normal_solver_metrics=False,
):
    sample_list = []
    for question in experience_question_list:
        sample = {
            'dataset': question['dataset'],
            'id': question['id'],
            'text': question['text'],
            'target': question['target'],
            'mmlu_subject': None if question['dataset'] != 'mmlu' else question['metadata']['subject'],
            'ceval_subject': None if question['dataset'] != 'ceval' else question['metadata']['subject'],
            'bbh_task_name': None if question['dataset'] != 'bbh' else question['metadata']['task_name'],
        }
        for solver_name in normal_solver_name_list:
            normal_sample = update_solver(question, sample, solver_name, average_normal_solver_metrics)
        for solver_name in maj_solver_name_list:
            update_solver_majority_vote(question, sample, solver_name)
        sample_list.append(sample)
    
    # drop null
    df = pd.DataFrame(sample_list)
    metrics_columns = []
    for metrics_name in ['accuracy', 'time_cost', 'dollar_cost']:
        for solver_name in normal_solver_name_list + maj_solver_name_list:
            metrics_columns.append(f"{solver_name}_{metrics_name}")
    invalid_indexer = df[metrics_columns].isnull().any(axis=1)
    print("{} abnormal samples are dropped!".format(len(df[invalid_indexer])))
    df = df[~invalid_indexer]

    # train test split
    rng = np.random.default_rng(12345)
    ret_list = []
    for _, row in df.iterrows():
        d = row.to_dict()
        # hard coded, only mmlu as the training set
        if d['dataset'] == 'mmlu':
            d['split'] = rng.choice(['TRAIN', 'DEV', 'TEST'], p=[train_ratio, dev_ratio, test_ratio])
        else:
            d['split'] = 'TEST'
        ret_list.append(d)
    df = pd.DataFrame(ret_list)
    return df