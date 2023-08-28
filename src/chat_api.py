from msal import PublicClientApplication, SerializableTokenCache
import json
import time
import os
import atexit
import requests
import tiktoken

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
        return [ret['message']['content'] for ret in j['choices']], request_time

 
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