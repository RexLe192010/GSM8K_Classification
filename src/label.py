from datasets import load_dataset, concatenate_datasets, Dataset
import pandas as pd
from src.utils import ChatGPT
import os
import random
import queue
import traceback
import concurrent.futures
import uuid
import tiktoken
import time
import copy
import pandas as pd
import threading
import fire
import jsonlines
from src.solver import get_solver
from collections import defaultdict
from src.data_utils import get_labeled_question_list
from src.utils import ChatGPTInternal

lock = threading.Lock()
random.seed(42)
pd.set_option('display.min_rows', 200)
pd.set_option('display.max_rows', 200)

def save(labeled_name, labeled_list):
    with jsonlines.open('./labeled_dataset/{}.jsonl'.format(labeled_name), 'w') as writer:
        for row in labeled_list:
            copied_row = copy.deepcopy(row)
            writer.write(copied_row)
    print("saved!")

def work(worker_id, q, limit_rpm, labeled_name, labeled_list, llm_type, internal_n_times, cnt):
    if llm_type == 'internal':
        llm = ChatGPTInternal(request_timeout=60, limit_rpm=limit_rpm, n_times=internal_n_times)
    elif llm_type == 'openai':
        llm = ChatGPT(request_timeout=60, limit_rpm=limit_rpm)
    else:
        raise NotImplementedError
    finished_cnt = 0
    while not(q.empty()):
        try:
            question = q.get() 
            solver_name = question['solver_info']['name']
            solver = get_solver(solver_name)
            if llm_type == 'openai':
                solver.solve(llm, question)
                labeled_list.append(question)
            elif llm_type == "internal":
                question_list = solver.solve(llm, question)
                labeled_list.extend(question_list)
            else:
                raise NotImplementedError
            finished_cnt += 1
            with lock:
                print('[worker_id {}] finished / all: {}/{}  Label list num {}'.format(
                    worker_id, finished_cnt, cnt, len(labeled_list))
                )
                if finished_cnt == 5 or finished_cnt % 10 == 0:
                    save(labeled_name, labeled_list)
        except Exception as e:
            traceback.print_exc()
            time.sleep(60)

def main(
    raw_name_list=["mmlu"], 
    labeled_name='mmlu_analyze_q',
    solver_name_to_n_tries={'analyze_q': 5},
    worker_num=1,
    limit_rpm_per_worker=10,
    llm_type='internal',
    internal_n_times=5,
):
    labeled_list = []
    labeled_set = set()
    id_solver_name_to_try_idx = defaultdict(int)
    save_path = './labeled_dataset/{}.jsonl'.format(labeled_name)
    if os.path.exists(save_path):
        for row in get_labeled_question_list(save_path):
            try_idx = id_solver_name_to_try_idx[row['id'], row['solver_info']['name']]
            row['solver_info']['try_idx'] = try_idx
            labeled_list.append(row)
            labeled_set.add((row['id'], row['solver_info']['name'], try_idx))
            id_solver_name_to_try_idx[row['id'], row['solver_info']['name']] = try_idx + 1

    q = queue.Queue()
    cnt = 0
    for raw_name in raw_name_list:
        with jsonlines.open('./raw_dataset/{}.jsonl'.format(raw_name)) as reader:
            for question_j in reader:
                for solver_name, n_tries in solver_name_to_n_tries.items():
                    prev_max_try_idx = -1
                    for try_idx in range(n_tries):
                        if (question_j['id'], solver_name, try_idx) in labeled_set:
                            prev_max_try_idx = try_idx
                    if prev_max_try_idx < n_tries - 1:
                        if llm_type == 'internal':
                            for start_try_idx in range(prev_max_try_idx + 1, n_tries, internal_n_times):
                                question = copy.deepcopy(question_j)
                                question['solver_info'] = {}
                                question['solver_info']['name'] = solver_name
                                question['solver_info']['start_try_idx'] = start_try_idx
                                q.put(question)
                                cnt += 1
                        elif llm == 'openai':
                            for try_idx in range(prev_max_try_idx + 1, n_tries):
                                question = copy.deepcopy(question_j)
                                question['solver_info'] = {}
                                question['solver_info']['name'] = solver_name
                                question['solver_info']['try_idx'] = start_try_idx
                                q.put(question)
                                cnt += 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_num) as executor:
        futures = [
            executor.submit(
                work, worker_id, q, limit_rpm_per_worker, labeled_name, labeled_list,
                llm_type, internal_n_times, cnt
            ) for worker_id in range(worker_num)
        ]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())
    save(labeled_name, labeled_list)


if __name__ == '__main__':
    fire.Fire(main)