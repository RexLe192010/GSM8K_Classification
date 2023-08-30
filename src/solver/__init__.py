from src.solver.llm_solver import DirectPromptSolver, CoTPromptSolver, GeneralPromptSolver
from src.solver.search_solver import SearchBasedSolver

def get_solver(solver_name):
    if solver_name == 'direct':
        return DirectPromptSolver()
    elif solver_name == 'cot':
        return CoTPromptSolver()
    elif solver_name == 'search':
        return SearchBasedSolver()
    elif solver_name == 'analyze_q':
        return GeneralPromptSolver(use_type='analyze_q')
    elif solver_name == 'analyze_v':
        return GeneralPromptSolver(use_type='analyze_c')
    else:
        raise NotImplementedError

