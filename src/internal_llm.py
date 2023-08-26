from msal import PublicClientApplication, SerializableTokenCache
import json
import os
import atexit
import requests

class LLMClient:

    _ENDPOINT = 'https://fe-26.qas.bing.net/completions'
    _SCOPES = ['api://68df66a4-cad9-4bfd-872b-c6ddde00d6b2/access']

    def __init__(self):
        self._cache = SerializableTokenCache()
        atexit.register(lambda: 
            open('.llmapi.bin', 'w').write(self._cache.serialize())
            if self._cache.has_state_changed else None)

        self._app = PublicClientApplication('68df66a4-cad9-4bfd-872b-c6ddde00d6b2', authority='https://login.microsoftonline.com/72f988bf-86f1-41af-91ab-2d7cd011db47', token_cache=self._cache)
        if os.path.exists('.llmapi.bin'):
            self._cache.deserialize(open('.llmapi.bin', 'r').read())

    def send_request(self, model_name, request):
        # get the token
        token = self._get_token()

        # populate the headers
        headers = {
            'Content-Type':'application/json', 
            'Authorization': 'Bearer ' + token, 
            'X-ModelType': model_name }

        body = str.encode(json.dumps(request))
        response = requests.post(LLMClient._ENDPOINT, data=body, headers=headers)
        return response.json()

    def send_stream_request(self, model_name, request):
        # get the token
        token = self._get_token()

        # populate the headers
        headers = {
            'Content-Type':'application/json', 
            'Authorization': 'Bearer ' + token, 
            'X-ModelType': model_name }

        body = str.encode(json.dumps(request))
        response = requests.post(LLMClient._ENDPOINT, data=body, headers=headers, stream=True)
        for line in response.iter_lines():
            text = line.decode('utf-8')
            if text.startswith('data: '):
                text = text[6:]
                if text == '[DONE]':
                    break
                else:
                    yield json.loads(text)       

    def _get_token(self):
        accounts = self._app.get_accounts()
        result = None

        if accounts:
            # Assuming the end user chose this one
            chosen = accounts[0]

            # Now let's try to find a token in cache for this account
            result = self._app.acquire_token_silent(LLMClient._SCOPES, account=chosen)

    
        if not result:
            # So no suitable token exists in cache. Let's get a new one from AAD.
            flow = self._app.initiate_device_flow(scopes=LLMClient._SCOPES)

            if "user_code" not in flow:
                raise ValueError(
                    "Fail to create device flow. Err: %s" % json.dumps(flow, indent=4))

            print(flow["message"])

            result = self._app.acquire_token_by_device_flow(flow)

        print(result)


        return result["access_token"]

prompt = '''
# Task

Your task is to solve the math problem, and put the number answer after ####.

# Example 1

Question:
Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer:
Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72

# Example 2
Question:
Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer:
Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. #### 10

# Example 3
Question:
Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Answer:
In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50. Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30. This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more. #### 5

# Output

Now, give you the following question:

Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?

Please generate a proper answer to this question. Put the final answer after ####.
'''.strip()

llm_client = LLMClient()

request_data = {
        "prompt": prompt,
        "max_tokens": 1536,
        "temperature": 0.6,
        "top_p":1,
        "n":5,
        "stream": False,
        "stop": None,
}

# Available models are listed here: https://msasg.visualstudio.com/QAS/_wiki/wikis/QAS.wiki/134728/Getting-Started-with-Substrate-LLM-API?anchor=available-models
response = llm_client.send_request('dev-gpt-35-turbo', request_data)
print(response)

