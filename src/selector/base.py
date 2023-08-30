import numpy as np

class SolverSelector(object):
    
    def __init__(self):
        pass

    def apply(self, df):
        pass

class NaiveSelector(SolverSelector):
    
    def __init__(self, accuracy_alpha, time_alpha, dollar_alpha, valid_solver_names, always_select):
        assert always_select in valid_solver_names
        self.always_select = always_select
    
    def apply(self, df):
        df['select_solver_name'] = self.always_select
        df['select_solver_name_confidence'] = 1.

class RandomSelector(SolverSelector):
    
    def __init__(self, accuracy_alpha, time_alpha, dollar_alpha, valid_solver_names, solver_name_to_prob=None):
        self.valid_solver_names = valid_solver_names
        if solver_name_to_prob is None:
            self.p_list = [1. / len(self.valid_solver_names) for solver_name in self.valid_solver_names]
        else:
            self.p_list = [solver_name_to_prob[solver_name] for solver_name in self.valid_solver_names]
        # must use a different seed from train/test/dev split!
        self.rng = np.random.default_rng(42)
    
    def apply(self, df):
        select_solver_name_list = []
        for _ in range(len(df)):
            select_solver_name_list.append(self.rng.choice(self.valid_solver_names, p=self.p_list))
        df['select_solver_name'] = select_solver_name_list
        df['select_solver_name_confidence'] = 1.


class OracleSelector(SolverSelector):
    
    def __init__(self, accuracy_alpha, time_alpha, dollar_alpha, valid_solver_names):
        self.accuracy_alpha = accuracy_alpha
        self.time_alpha = time_alpha
        self.dollar_alpha = dollar_alpha
        self.valid_solver_names = valid_solver_names
    
    def apply(self, df):
        
        def get_oracle_solver_name(row):
            solver_name_info_list = []
            for solver_name in self.valid_solver_names:
                utility = 0
                # use oracle here
                utility += self.accuracy_alpha * row[f"{solver_name}_accuracy"]
                utility += -self.time_alpha * row[f"{solver_name}_time_cost"]
                utility += -self.dollar_alpha * row[f"{solver_name}_dollar_cost"]
                solver_name_info_list.append((utility, solver_name))
            solver_name_info_list = sorted(solver_name_info_list, reverse=True)
            select_solver_name = solver_name_info_list[0][1]
            return select_solver_name
        df['select_solver_name'] = df.apply(get_oracle_solver_name, axis=1)
        df['select_solver_name_confidence'] = 1.0