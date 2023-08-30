from src.solver.base import Solver
from src.solver.llm_solver import mmlu_direct_prompt_template, bb_direct_prompt_template, get_choices_text
import jsonlines
import copy
import requests
import time

whether_search_prompt_template = \
'''
# Task

Your task is to determine whether we should do a web search to answer an input question.
If it is necessary to carry out a web search, output the query words;
If not, output N/A for the query words.

# Examples

# Example 1

Question: The most rapidly urbanizing area of the world is:
Choices:
A. Europe
B. East Asia
C. Sub-Saharan Africa
D. South Asia

Reasoning: It is a question about fact checking and it is necessary to use a web search.
Query Words: the most rapidly urbanizing area of the world.

# Example 2

Question: In what way were Neandertals physically different from modern Homo sapiens?
Choices:
A. Neandertals had wide, squat torsos and short extremities.
B. Neandertals had smaller brains and larger noses.
C. Neandertal skeletons have more bones that modern Homo sapiens.
D. Both a and b.

Reasoning: It is a question about prehistory knowledge and it is necessary to use a web search.
Query Words: differences between Neandertals and modern Homo sapiens

# Example 3

Question: Approximately how old is the surface of Venus?
Choices:
A. 750 million years.
B. 2 billion years.
C. 3 billion years.
D. 4.5 billion years.

Reasoning: It is a question about astronomy knowledge and it is necessary to use a web search.
Query Words: differences between Neandertals and modern Homo sapiens.

# Example 4

Question: As of 2017, the share of deaths in Greenland by suicide is about
Choices:
A. 0.90%
B. 1.80%
C. 3.60%
D. 7.20%

Reasoning: It is a question about fact checking and it is necessary to use a web search.
Query Words: the share of deaths in Greenland by suicide 2017

# Example 5

Question: Last year, Chesa made 32 one-cup servings of soup for a school party. This year, she will make two times the amount of soup that she made last year. How many gallons of soup will Chesa make this year?
Choices:
A. 64
B. 16
C. 4
D. 2

Reasoning: This question doesn't need external knowledge.
Query Words: N/A

# Input Question

{question_text}

# Output Format

Reasoning: <short reasoning to justify whether a web search is necessary for the input question>
Query Words: <query words for the search engine; use N/A if web search is not necessary>
'''.strip()

one_organic_text_template = '''
Title: {title}
Snippet: {snippet}
'''.strip()

one_qa_text_template = '''
Question: {question}
Answer: {answer}
'''.strip()

web_search_text_template = '''
{qa_text}

{organic_text}
'''.strip()

mmlu_direct_with_search_prompt_template = '''
Here's some background knowledge that might be useful from web search:

{web_search_text}

Now, you are given the following question:

{question_text}

Output your answer directly ({choices_text}) without adding anything else.
'''.strip()

bb_direct_with_search_prompt_template = '''
Here's some background knowledge that might be useful from web search:

{web_search_text}

Now, you are given the following question:

{question_text}

Output your answer directly without adding anything else.
'''.strip()

class SearchBasedSolver(Solver):
    
    def prepare_question(self, question):
        super(SearchBasedSolver, self).prepare_question(question)
        question['solver_info']['whether_search_time_cost'] = 0
        question['solver_info']['whether_search_prompt_token_num'] = 0
        question['solver_info']['whether_search_response_token_num'] = 0
        question['solver_info']['whether_search_dollar_cost'] = 0
        question['solver_info']['whether_search_raw_response'] = 0
        question['solver_info']['whether_search'] = False

        question['solver_info']['search_time_cost'] = 0
        question['solver_info']['search_dollar_cost'] = 0
        question['solver_info']['search_words'] = None
        question['solver_info']['web_seach_text'] = 0

        question['solver_info']['final_ans_time_cost'] = 0
        question['solver_info']['final_ans_raw_response'] = 0
        question['solver_info']['final_ans_prompt_token_num'] = 0
        question['solver_info']['final_ans_response_token_num'] = 0
        question['solver_info']['final_ans_dollar_cost'] = 0

    
    def solve(self, llm, question):
        # step 1: determine whether to search
        self.prepare_question(question)
        prompt = whether_search_prompt_template.format(question_text=question['text'])
        question['solver_info']['whether_search_prompt_token_num'] = len(llm.enc.encode(prompt))
        whether_search_raw_response, whether_search_time_cost = llm(prompt)
        question['solver_info']['whether_search_response_token_num'] = len(llm.enc.encode(whether_search_raw_response))
        question['solver_info']['whether_search_raw_response'] = whether_search_raw_response
        question['solver_info']['whether_search_time_cost'] = whether_search_time_cost
        question['solver_info']['whether_search_dollar_cost'] = \
            (0.0015 * question['solver_info']['whether_search_prompt_token_num'] + 0.002 * question['solver_info']['whether_search_response_token_num']) / 1000
        pos_of_search_words = whether_search_raw_response.lower().find('query words:')
        whether_search = False 
        search_words = None
        if pos_of_search_words != -1:
            search_words = whether_search_raw_response[pos_of_search_words + len('query words:'):].strip()
            if search_words.lower() != 'n/a':
                whether_search = True
        question['solver_info']['whether_search'] = whether_search
        
        # step 2: do the search
        if whether_search:
            question['solver_info']['search_words'] = search_words
            params = {
                'api_key': 'BE0B6C798ACB4DDBABBD6D49E8058520',
                'q': search_words,
                'location': 'United States',
                'google_domain': 'google.com',
                'gl': 'us',
                'hl': 'en'
            }
            search_time_start = time.time()
            api_result = requests.get('https://api.valueserp.com/search', params, timeout=60)
            j = api_result.json()
            search_time_cost = time.time() - search_time_start 
            question['solver_info']['search_time_cost'] = search_time_cost
            question['solver_info']['search_dollar_cost'] = 2.5 / 1000
            assert j['request_info']['success']
            if 'questions_and_answers' in j:
                qa_text_list = []
                for ret in j['questions_and_answers'][:4]:
                    qa_text_list.append(one_qa_text_template.format(**ret))
                qa_text = '\n\n'.join(qa_text_list)
            else:
                qa_text = ''
            organic_text_list = []
            for ret in j['organic_results'][:6]:
                if 'snippet' in ret and 'title' in ret:
                    organic_text_list.append(one_organic_text_template.format(**ret))
            organic_text = '\n\n'.join(organic_text_list)

            web_search_text = web_search_text_template.format(qa_text=qa_text, organic_text=organic_text).strip()
            question['solver_info']['web_seach_text'] = web_search_text

        # step 3: get the ans
        if whether_search:
            if question['dataset'] == 'mmlu':
                prompt = mmlu_direct_with_search_prompt_template.format(
                    question_text=question['text'],
                    web_search_text=web_search_text,
                    choices_text=get_choices_text(question['choice_num']),
                )
            elif question['dataset'] == 'bbh':
                prompt = bb_direct_with_search_prompt_template.format(
                    question_text=question['text'],
                    web_search_text=web_search_text,
                )
            else:
                raise NotImplementedError
        else:
            if question['dataset'] == 'mmlu':
                prompt = mmlu_direct_prompt_template.format(
                    question_text=question['text'],
                    choices_text=get_choices_text(question['choice_num']),
                )
            elif question['dataset'] == 'bbh':
                prompt = bb_direct_prompt_template.format(
                    question_text=question['text'],
                )
            else:
                raise NotImplementedError
        question['solver_info']['final_ans_prompt_token_num'] += len(llm.enc.encode(prompt))
        final_ans_raw_response, final_ans_time_cost = llm(prompt)
        question['solver_info']['final_ans_response_token_num'] += len(llm.enc.encode(final_ans_raw_response))
        question['solver_info']['final_ans_raw_response'] = final_ans_raw_response
        question['solver_info']['final_ans_time_cost'] = final_ans_time_cost
        question['solver_info']['final_ans_dollar_cost'] = \
            (0.0015 * question['solver_info']['final_ans_prompt_token_num'] + 0.002 * question['solver_info']['final_ans_response_token_num']) / 1000
        solver_ans = solver_ans = self.postprocess(question, final_ans_raw_response)
        self.examine_answer(question, solver_ans)
        question['solver_info']['time_cost'] = question['solver_info']['whether_search_time_cost'] + \
            question['solver_info']['search_time_cost'] + \
            question['solver_info']['final_ans_time_cost']
        question['solver_info']['dollar_cost'] = question['solver_info']['whether_search_dollar_cost'] + \
            question['solver_info']['search_dollar_cost'] + \
            question['solver_info']['final_ans_dollar_cost']
            