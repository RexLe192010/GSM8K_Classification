import time
import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import tiktoken
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from langchain.embeddings import OpenAIEmbeddings
import os
import hashlib
import pandas as pd
import copy
from functools import cmp_to_key
from msal import PublicClientApplication, SerializableTokenCache
import json
import atexit
import requests

class ChatGPT(object):

    def __init__(self, temp=0.7, request_timeout=60, limit_rpm=60):
        # Warning: Not Support Multi-thread
        self.chat = ChatOpenAI(
            temperature=temp,
            request_timeout=request_timeout
        )
        self.limit_rpm = int(limit_rpm)
        self.request_ts_list = []
        self.p = 0
        self.enc = tiktoken.get_encoding('cl100k_base')

    def call_chat(self, *args, **kwargs):
        cur_time = time.time()
        self.request_ts_list.append(cur_time)
        while not(self.request_ts_list[self.p] >= cur_time - 60):
            self.p += 1
        considered_start_time = self.request_ts_list[self.p]
        if considered_start_time != cur_time and (len(self.request_ts_list) - self.p) >= self.limit_rpm:
            # need wait
            wait_time = max(self.request_ts_list[len(self.request_ts_list) - self.limit_rpm] + 60 - cur_time + 0.5, 0)
            time.sleep(wait_time)
            self.request_ts_list[-1] = time.time()

        start_time = time.time()
        ret = self.chat(*args, **kwargs)
        request_time = time.time() - start_time
        return ret.content, request_time


    def __call__(self, prompt):
        messages = [
            SystemMessage(content="You are an helpful assistant."),
            HumanMessage(content=prompt)
        ]
        return self.call_chat(messages)

    def multi_round_call(self, conversation_list, human_first=True):
        messages = [
            SystemMessage(content="You are an helpful assistant."),
        ]
        if human_first:
            diff = 0
        else:
            diff = 1
        for idx, conversation_text in enumerate(conversation_list):
            if (diff + idx) % 2 == 0:
                messages.append(HumanMessage(content=conversation_text))
            else:
                messages.append(AIMessage(content=conversation_text))
        return self.call_chat(messages)


class MPNet(object):

    def __init__(self):
        self.model = None

    def inference(self, messages):
        if self.model is None:
            self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        assert isinstance(messages, list)
        return self.model.encode(messages).tolist()


def ensure_text_embedding(df, batch_size=128):
    if 'text_embedding' in df.columns:
        return
    text_embedding_list = []
    mpnet = MPNet()
    batched_text_list = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        batched_text_list.append(row.text)
        if len(batched_text_list) == batch_size:
            text_embedding_list.extend(mpnet.inference(batched_text_list))
            batched_text_list = []
    if len(batched_text_list) > 0:
        text_embedding_list.extend(mpnet.inference(batched_text_list))
        batched_text_list = []
    df['text_embedding'] = text_embedding_list


def ensure_openai_text_embedding(df, batch_size=128, cache_path='raw_dataset/openai_embedding_cache.pkl'):
    if 'openai_text_embedding' in df.columns:
        return

    md5sum_to_embedding = {}
    if os.path.exists(cache_path):
        cache_info_df = pd.read_pickle(cache_path)
        for _, row in cache_info_df.iterrows():
            md5sum_to_embedding[row.md5sum] = list(row.embedding)

    embeddings_model = OpenAIEmbeddings(model='text-embedding-ada-002')
    batched_text_list = []
    batched_md5sum_list = []
    skip_cnt = 0
    run_cnt = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        md5sum = hashlib.md5(row.text.encode('utf-8')).hexdigest()
        if md5sum in md5sum_to_embedding:
            skip_cnt += 1
        else:
            batched_text_list.append(row.text)
            batched_md5sum_list.append(md5sum)
            if len(batched_text_list) == batch_size:
                embeddings = embeddings_model.embed_documents(batched_text_list)
                for md5sum, embedding in zip(batched_md5sum_list, embeddings):
                    md5sum_to_embedding[md5sum] = embedding
                    run_cnt += 1
                batched_text_list = []
                batched_md5sum_list = []
    if len(batched_text_list) > 0:
        embeddings = embeddings_model.embed_documents(batched_text_list)
        for md5sum, embedding in zip(batched_md5sum_list, embeddings):
            md5sum_to_embedding[md5sum] = embedding
            run_cnt += 1
        batched_text_list = []
        batched_md5sum_list = []

    cache_df_row_list = []
    for md5sum, embedding in md5sum_to_embedding.items():
        cache_df_row_list.append({
            'md5sum': md5sum,
            'embedding': embedding
        })
    cache_df = pd.DataFrame(cache_df_row_list)
    cache_df.to_pickle(cache_path)
    print("OpenAI embedding: run {} times, skip {} times.".format(run_cnt, skip_cnt))

    text_embedding_list = []
    for _, row in df.iterrows():
        md5sum = hashlib.md5(row.text.encode('utf-8')).hexdigest()
        embedding = md5sum_to_embedding[md5sum]
        text_embedding_list.append(embedding)
    df['openai_text_embedding'] = text_embedding_list


class PairCount(object):
    def __init__(self, reverse=False):
        self.reverse = reverse
     
    def solve(self, nums):
        self.cnt = 0
        def merge(nums, start, mid, end, temp):
            i, j = start, mid + 1
            while i <= mid and j <= end:
                if (self.reverse and nums[i] <= nums[j]) or (not(self.reverse) and nums[i] >= nums[j]):
                    temp.append(nums[i])
                    i += 1
                else:
                    self.cnt += mid - i + 1
                    temp.append(nums[j])
                    j += 1
            while i <= mid:
                temp.append(nums[i])
                i += 1
            while j <= end:
                temp.append(nums[j])
                j += 1
             
            for i in range(len(temp)):
                nums[start + i] = temp[i]
            temp.clear()
                     
 
        def mergeSort(nums, start, end, temp):
            if start >= end: return
            mid = (start + end) >> 1
            mergeSort(nums, start, mid, temp)
            mergeSort(nums, mid + 1, end, temp)
            merge(nums, start, mid,  end, temp)
 
        mergeSort(nums, 0, len(nums) - 1, [])
        return self.cnt
 
 
def count_all_pair(label_list):
    # 我们需要在 label_list 中选择 i, j, 并且 label_list[i] > label_list[j]
    # 这样一共有多少对。
    all_pair_num = 0
    last_larger_count = 0
    # 排序一遍
    label_list = sorted(label_list)
    for i in range(len(label_list) - 2, -1, -1):
        if label_list[i] == label_list[i + 1]:
            all_pair_num += last_larger_count
        else:
            # label_list[i] < label[i + 1]
            all_pair_num += len(label_list) - i - 1
            last_larger_count = len(label_list) - i - 1
    return all_pair_num
 
def count_auc_reverse_pair(label_list):
    # 在 label_list 数逆序对的个数
    # 逆序对是严格的，即必须 label_list[i] > label_list[j]，
    # label_list[i] = label_list[j] 的情况不算
    strict_reverse_count = PairCount(True).solve(copy.copy(label_list))
    strict_ascending_count = PairCount(False).solve(copy.copy(label_list))
    strict_equal_count = len(label_list) * (len(label_list) - 1) / 2 - strict_reverse_count - strict_ascending_count
    return strict_equal_count * 0.5 + strict_reverse_count
 
def regression_auc_direct(y_true, y_pred):
    def compare(row1, row2):
        if abs(row1[0] - row2[0]) < 1e-8 and abs(row1[1] - row2[1]) < 1e-8:
            return 0
        if abs(row1[0] - row2[0]) < 1e-8 and row1[1] < row2[1]:
            return -1
        if row1[0] < row2[0]:
            return -1
        return 1
    sorted_y = sorted(zip(y_true, y_pred), key=cmp_to_key(compare))
    label_list = [item[0] for item in sorted_y]
    prediction_list = [item[1] for item in sorted_y]
    all_pair_num = count_all_pair(label_list)
    if all_pair_num == 0:
        return None
    reverse_pair_num = count_auc_reverse_pair(prediction_list)
    equal_combo_num = 1
    equal_pair = 0
    for i in range(1, len(sorted_y)):
        if abs(sorted_y[i][0] - sorted_y[i - 1][0]) < 1e-8 and abs(sorted_y[i][1] - sorted_y[i - 1][1]) < 1e-8:
            equal_combo_num += 1
        else:
            equal_combo_num = 1
        equal_pair = equal_pair + (equal_combo_num - 1)
    reverse_pair_num -= equal_pair * 0.5
    return (all_pair_num - reverse_pair_num) / all_pair_num


class ChatGPTInternal(object):

    _ENDPOINT = 'https://fe-26.qas.bing.net/completions'
    _SCOPES = ['api://68df66a4-cad9-4bfd-872b-c6ddde00d6b2/access']

    def __init__(self, temp=0.7, request_timeout=60, limit_rpm=60, max_tokens=2048, n_times=1):
        self._cache = SerializableTokenCache()
        atexit.register(lambda: 
            open('.llmapi.bin', 'w').write(self._cache.serialize())
            if self._cache.has_state_changed else None)

        self._app = PublicClientApplication('68df66a4-cad9-4bfd-872b-c6ddde00d6b2', authority='https://login.microsoftonline.com/72f988bf-86f1-41af-91ab-2d7cd011db47', token_cache=self._cache)
        if os.path.exists('.llmapi.bin'):
            self._cache.deserialize(open('.llmapi.bin', 'r').read())

        self.request_timeout = request_timeout
        self.temp = temp
        self.limit_rpm = int(limit_rpm)
        self.request_ts_list = []
        self.p = 0
        self.enc = tiktoken.get_encoding('cl100k_base')
        self.max_tokens = max_tokens
        self.n_times = n_times

    def call_completion(self, prompt):
        model_name = "dev-gpt-35-turbo"
        request_data = {
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temp,
            "top_p": 1,
            "n": self.n_times,
            "stream": False,
            "stop": None
        }
        return self.send_request(model_name, request_data)
        

    def call_chat(self, prompt):
        model_name = "dev-chat-completion-gpt-35-turbo"
        request_data = {
            "messages": [
                {"role": "system", "content": "You are an helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temp,
            "n": self.n_times,
            "stream": False,
            "stop": None,
        }
        return self.send_request(model_name, request_data)


    def send_request(self, model_name, request):
        # get the token
        token = self._get_token()
         # populate the headers
        headers = {
            'Content-Type':'application/json', 
            'Authorization': 'Bearer ' + token, 
            'X-ModelType': model_name 
        }
        body = str.encode(json.dumps(request))


        cur_time = time.time()
        self.request_ts_list.append(cur_time)
        while not(self.request_ts_list[self.p] >= cur_time - 60):
            self.p += 1
        considered_start_time = self.request_ts_list[self.p]
        if considered_start_time != cur_time and (len(self.request_ts_list) - self.p) >= self.limit_rpm:
            # need wait
            wait_time = max(self.request_ts_list[len(self.request_ts_list) - self.limit_rpm] + 60 - cur_time + 0.5, 0)
            time.sleep(wait_time)
            self.request_ts_list[-1] = time.time()

        start_time = time.time()
        response = requests.post(ChatGPTInternal._ENDPOINT, data=body, headers=headers, timeout=self.request_timeout)
        request_time = time.time() - start_time
        j = response.json()
        if 'error' in j:
            print(j)
            # trigger an auto retry
            time.sleep(62)
            return self.send_request(model_name, request)
        # print(j)
        return j, request_time

    def _get_token(self):
        accounts = self._app.get_accounts()
        result = None

        if accounts:
            # Assuming the end user chose this one
            chosen = accounts[0]

            # Now let's try to find a token in cache for this account
            result = self._app.acquire_token_silent(ChatGPTInternal._SCOPES, account=chosen)

    
        if not result:
            # So no suitable token exists in cache. Let's get a new one from AAD.
            flow = self._app.initiate_device_flow(scopes=ChatGPTInternal._SCOPES)

            if "user_code" not in flow:
                raise ValueError(
                    "Fail to create device flow. Err: %s" % json.dumps(flow, indent=4))

            print(flow["message"])

            result = self._app.acquire_token_by_device_flow(flow)

        return result["access_token"]