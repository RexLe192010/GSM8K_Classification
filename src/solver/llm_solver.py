import traceback
from src.solver.base import Solver
import copy

mmlu_direct_prompt_template = '''
You are given the following question:

{question_text}

Output your answer directly ({choices_text}) without adding anything else.
'''.strip()

bb_direct_prompt_template = '''
You are given the following question:

{question_text}

Output your answer directly without adding anything else.
'''.strip()

ceval_direct_prompt_template = '''
下面是一道单项选择题：

{question_text}

直接输出你的答案（{choices_text}），不要添加其他任何内容。
'''.strip()

mmlu_cot_prompt_template = '''
# Task
You are given the following question:

{question_text}

# Output Format

Reasoning: <Think it step by step and output the reasoning>
Answer: <Choose from {choices_text}>
'''.strip()

bb_cot_prompt_template = '''
You are given the following question:

{question_text}

# Output Format

Reasoning: <Think it step by step and output the reasoning>
Answer: <Place the answer here without adding anything else>
'''.strip()

ceval_cot_prompt_template = '''
下面是一道单项选择题：

{question_text}

# 输出格式

推理：<一步一步思考并输出推理过程>
答案：<在 {choices_text} 中选择一个>
'''.strip()


mmlu_analyze_q_prompt_template = '''
# Task
You are given the following question:

{question_text}

# Output Format

Analysis: <Analyze the question stem carefully and output the analysis>
Answer: <Choose from {choices_text}>
'''.strip()

mmlu_analyze_c_prompt_template = '''
# Task
You are given the following question:

{question_text}

# Output Format

Analysis: <Analyze the choices one by one and output the analysis>
Answer: <Choose from {choices_text}>
'''.strip()


def get_choices_text(num):
    choice_list = []
    for idx in range(num):
        choice_list.append(chr(ord('A') + idx))
    return '/'.join(choice_list)


class DirectPromptSolver(Solver):

    def __init__(self):
        pass

    def prepare_question(self, question):
        super(DirectPromptSolver, self).prepare_question(question)
        question['solver_info']['prompt_token_num'] = 0
        question['solver_info']['response_token_num'] = 0
        question['solver_info']['raw_response'] = None

    def solve(self, llm, question):
        self.prepare_question(question)
        solver_ans = None
        time_cost = None
        dollar_cost = None
        if question['dataset'] == 'mmlu':
            choices_text = get_choices_text(question['choice_num'])
            prompt = mmlu_direct_prompt_template.format(
                choices_text=choices_text,
                question_text=question["text"]
            )
        elif question['dataset'] == 'ceval':
            choices_text = get_choices_text(question['choice_num'])
            prompt = ceval_direct_prompt_template.format(
                choices_text=choices_text,
                question_text=question["text"]
            )
        elif question['dataset'] == 'bbh':
            prompt = bb_direct_prompt_template.format(
                question_text=question["text"]
            )
        else:
            raise NotImplementedError
        question['solver_info']['prompt_token_num'] = len(llm.enc.encode(prompt))
        raw_response, time_cost = llm(prompt)
        question['solver_info']['raw_response'] = raw_response
        question['solver_info']['response_token_num'] = len(llm.enc.encode(raw_response))
        dollar_cost = (0.0015 * question['solver_info']['prompt_token_num'] + 0.002 * question['solver_info']['response_token_num']) / 1000
        solver_ans = self.postprocess(question, raw_response)
        self.examine_answer(question, solver_ans)
        question['solver_info']['time_cost'] = time_cost
        question['solver_info']['dollar_cost'] = dollar_cost
        return question


class CoTPromptSolver(Solver):

    def __init__(self):
        pass

    def prepare_question(self, question):
        super(CoTPromptSolver, self).prepare_question(question)
        question['solver_info']['prompt_token_num'] = 0
        question['solver_info']['response_token_num'] = 0
        question['solver_info']['raw_response'] = None

    def solve(self, llm, question):
        self.prepare_question(question)
        solver_ans = None
        time_cost = None
        dollar_cost = None
        if question['dataset'] == 'mmlu':
            choices_text = get_choices_text(question['choice_num'])
            prompt = mmlu_cot_prompt_template.format(
                choices_text=choices_text,
                question_text=question["text"]
            )
        elif question['dataset'] == 'ceval':
            choices_text = get_choices_text(question['choice_num'])
            prompt = ceval_cot_prompt_template.format(
                choices_text=choices_text,
                question_text=question["text"]
            )
        elif question['dataset'] == 'bbh':
            prompt = bb_cot_prompt_template.format(
                question_text=question["text"]
            )
        else:
            raise NotImplementedError
        question['solver_info']['prompt_token_num'] = len(llm.enc.encode(prompt))
        raw_response, time_cost = llm(prompt)
        question['solver_info']['raw_response'] = raw_response
        question['solver_info']['response_token_num'] = len(llm.enc.encode(raw_response))
        dollar_cost = (0.0015 * question['solver_info']['prompt_token_num'] + 0.002 * question['solver_info']['response_token_num']) / 1000
        if question['dataset'] == 'ceval':
            response = raw_response
            ans_pos = response.rfind('答案:')
            response = raw_response[ans_pos + len("答案:"):].strip()
            ans_pos = response.rfind('答案：')  # different ":"
            response = response[ans_pos + len("答案："):].strip()
        else:
            ans_pos = raw_response.rfind('Answer:')
            response = raw_response[ans_pos + len("Answer:"):].strip()
        solver_ans = self.postprocess(question, response)
        self.examine_answer(question, solver_ans)
        question['solver_info']['time_cost'] = time_cost
        question['solver_info']['dollar_cost'] = dollar_cost
        return question


class GeneralPromptSolver(Solver):

    def __init__(self, use_internal_gpt=True, use_type='analyze_q'):
        self.use_internal_gpt = use_internal_gpt
        self.use_type = use_type

    def solve(self, llm, question):
        assert self.use_internal_gpt is True
        self.prepare_question(question)
        solver_ans = None
        time_cost = None
        dollar_cost = None
        if question['dataset'] == 'mmlu':
            choices_text = get_choices_text(question['choice_num'])
            if self.use_type == 'analyze_q':
                prompt = mmlu_analyze_q_prompt_template.format(
                    choices_text=choices_text,
                    question_text=question["text"]
                )
            elif self.use_type == 'analyze_c':
                prompt = mmlu_analyze_c_prompt_template.format(
                    choices_text=choices_text,
                    question_text=question["text"]
                )
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        question['solver_info']['prompt_token_num'] = len(llm.enc.encode(prompt))
        ret, time_cost = llm.call_chat(prompt)
        try_list = []
        for choice in ret['choices']:
            one_question_try = copy.deepcopy(question)
            one_question_try['solver_info']['time_cost'] = time_cost
            raw_response = choice['message']['content']
            self.parse_raw_response(llm, one_question_try, raw_response)
            try_list.append(one_question_try)
        return try_list

    def parse_raw_response(self, llm, question, raw_response):
        question['solver_info']['raw_response'] = raw_response
        question['solver_info']['response_token_num'] = len(llm.enc.encode(raw_response))
        dollar_cost = (0.0015 * question['solver_info']['prompt_token_num'] + 0.002 * question['solver_info']['response_token_num']) / 1000
        ans_pos = raw_response.rfind('Answer:')
        response = raw_response[ans_pos + len("Answer:"):].strip()
        solver_ans = self.postprocess(question, response)
        self.examine_answer(question, solver_ans)
        question['solver_info']['dollar_cost'] = dollar_cost
        return question