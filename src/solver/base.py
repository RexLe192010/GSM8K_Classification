def get_choice_list(num):
    choice_list = []
    for idx in range(num):
        choice_list.append(chr(ord('A') + idx))
    return choice_list


class Solver(object):

    def prepare_question(self, question):
        question['solver_info']['ans'] = None
        question['solver_info']['is_true'] = False
        question['solver_info']['is_wrong'] = False
        question['solver_info']['is_failed'] = False
        question['solver_info']['time_cost'] = 0
        question['solver_info']['dollar_cost'] = 0

    def examine_answer(self, question, solver_ans):
        question['solver_info']['ans'] = solver_ans
        if question ['dataset'] == 'mmlu' or question ['dataset'] == 'ceval':
            choice_list = get_choice_list(question['choice_num'])
            if solver_ans in choice_list:
                if solver_ans == question['target']:
                    question['solver_info']['is_true'] = True
                else:
                    question['solver_info']['is_wrong'] = True
            else:
                question['solver_info']['is_failed'] = True
        elif question['dataset'] == 'bbh':
            if solver_ans == question['target']:
                question['solver_info']['is_true'] = True
            else:
                question['solver_info']['is_wrong'] = True
        else:
            raise NotImplementedError
        

    def mmlu_postprocess(self, question, raw_response):
        return raw_response.split()[0].replace('.', '').replace(',', '').replace(' ', '').strip().upper()[0]

    def ceval_postprocess(self, question, raw_response):
        ans = raw_response.split()[0].replace('.', '').replace(',', '')\
            .replace('。', '').replace('，', '')\
            .replace(' ', '').strip().upper()[0]
        if ans not in ['A', 'B', 'C', 'D']:
            pos_info_list = []
            for ans in ['A', 'B', 'C', 'D']:
                pos = raw_response.rfind(ans)
                if pos != -1:
                    pos_info_list.append((pos, ans))
            if len(pos_info_list) == 0:
                ans = None
            else:
                ans = sorted(pos_info_list, reverse=True)[0][1]
        return ans

    def bbh_postprocess(self, question, raw_response):
        if question['metadata']['task_name'] == 'word_sorting' or question['metadata']['task_name'] == 'dyck_languages':
            # these two tasks have space in the answer, so we do not do the split!
            solver_ans = raw_response.replace('.', '').replace(',', '').strip()
        else:
            solver_ans = raw_response.split()[0].replace('.', '').replace(',', '').replace(' ', '').strip()
        return solver_ans

    def postprocess(self, question, raw_response):
        if question['dataset'] == 'mmlu':
            return self.mmlu_postprocess(question, raw_response)
        elif question['dataset'] == 'bbh':
            return self.bbh_postprocess(question, raw_response)
        elif question['dataset'] == 'ceval':
            return self.ceval_postprocess(question, raw_response)
        else:
            raise NotImplementedError