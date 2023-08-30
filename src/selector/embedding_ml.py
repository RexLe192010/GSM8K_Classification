from src.selector.base import SolverSelector
from src.selector.class_utils import MLRegressionUtils, MLClassificationUtils, TreeUtils, EmbeddingRowFeatureUtils
from src.utils import ensure_text_embedding
import pandas as pd
from functools import partial
import numpy as np

class EmbeddingMLMultiTargetRegressionSelector(SolverSelector, EmbeddingRowFeatureUtils, MLRegressionUtils):
    
    def __init__(
        self, 
         accuracy_alpha, time_alpha, dollar_alpha, valid_solver_names, 
         ml_model_cls, ml_model_kwargs={}, enable_X_norm=True, enable_y_norm=True, embedding_type='mpnet',
    ):
        EmbeddingRowFeatureUtils.__init__(self, embedding_type)
        MLRegressionUtils.__init__(self, ml_model_cls, ml_model_kwargs, enable_X_norm, enable_y_norm)
        self.valid_solver_names = valid_solver_names
        assert len(self.valid_solver_names) != 1
        self.accuracy_alpha = accuracy_alpha
        self.time_alpha = time_alpha
        self.dollar_alpha = dollar_alpha
    
    def apply(self, df):
        self.ensure_text_embedding(df)
        model_metrics_name_list = []
        if self.accuracy_alpha != 0:
            model_metrics_name_list.append('accuracy')
        if self.time_alpha != 0:
            model_metrics_name_list.append('time_cost')
        if self.dollar_alpha != 0:
            model_metrics_name_list.append('dollar_cost')
        for solver_name in self.valid_solver_names:
            for metrics_name in model_metrics_name_list:
                print("Build regression model for {}_{}".format(solver_name, metrics_name))
                self.fit_and_transform(df, '{}_{}'.format(solver_name, metrics_name))
        def select_solver_and_get_confidence(row):
            solver_name_info_list = []
            for solver_name in self.valid_solver_names:
                utility = 0
                if self.accuracy_alpha != 0:
                    utility += self.accuracy_alpha * row[f"pred_{solver_name}_accuracy"]
                if self.time_alpha != 0:
                    utility += -self.time_alpha * row[f"pred_{solver_name}_time_cost"]
                if self.dollar_alpha != 0:
                    utility += -self.dollar_alpha * row[f"pred_{solver_name}_dollar_cost"]
                solver_name_info_list.append((utility, solver_name))
            solver_name_info_list = sorted(solver_name_info_list, reverse=True)
            select_solver_name = solver_name_info_list[0][1]
            select_solver_name_confidence = solver_name_info_list[0][0] - solver_name_info_list[1][0]
            return pd.Series([select_solver_name, select_solver_name_confidence], index=[
                'select_solver_name', 'select_solver_name_confidence'
            ])
        temp_df = df.apply(select_solver_and_get_confidence, axis=1)
        df['select_solver_name'] = temp_df['select_solver_name']
        df['select_solver_name_confidence'] = temp_df['select_solver_name_confidence']

# this selector only supports binary classes
class EmbeddingMLAccBinaryRegressionSelector(SolverSelector, EmbeddingRowFeatureUtils, MLRegressionUtils):
    
    def __init__(
        self, 
         accuracy_alpha, time_alpha, dollar_alpha, valid_solver_names, 
         ml_model_cls, ml_model_kwargs={}, enable_X_norm=True, enable_y_norm=True, embedding_type='mpnet',
    ):
        EmbeddingRowFeatureUtils.__init__(self, embedding_type)
        MLRegressionUtils.__init__(self, ml_model_cls, ml_model_kwargs, enable_X_norm, enable_y_norm)
        self.valid_solver_names = valid_solver_names
        assert len(self.valid_solver_names) == 2
        self.accuracy_alpha = accuracy_alpha
        self.time_alpha = time_alpha
        self.dollar_alpha = dollar_alpha

    def apply(self, df):
        self.ensure_text_embedding(df)
        solver_name_a = self.valid_solver_names[0]
        solver_name_b = self.valid_solver_names[1]
        # Model Accuracy Diff
        df['acc_diff'] = df['{}_accuracy'.format(solver_name_a)] - df['{}_accuracy'.format(solver_name_b)]
        self.fit_and_transform(df, 'acc_diff')
        # Predict Time/Dollar use TRAIN data only!
        estimate_time_diff = (
            df[(df.split == 'TRAIN')]['{}_time_cost'.format(solver_name_a)] - \
            df[(df.split == 'TRAIN')]['{}_time_cost'.format(solver_name_b)]
        ).mean()
        estimate_dollar_diff = (
            df[(df.split == 'TRAIN')]['{}_dollar_cost'.format(solver_name_a)] - \
            df[(df.split == 'TRAIN')]['{}_dollar_cost'.format(solver_name_b)]
        ).mean()
        df['pred_time_diff'.format(solver_name)] = estimate_time_diff
        df['pred_dollar_diff'.format(solver_name)] = estimate_dollar_diff
        select_solver_name_list = []
        select_solver_name_confidence_list = []
        for _, row in df.iterrows():
            pred_utility_diff = 0
            pred_utility_diff += self.accuracy_alpha * row.pred_acc_diff  # a - b
            pred_utility_diff += - self.time_alpha * row.pred_time_diff 
            pred_utility_diff += - self.dollar_alpha * row.pred_dollar_diff
            select_solver_name_confidence_list.append(pred_utility_diff)
            if pred_utility_diff > 0:
                select_solver_name_list.append(solver_name_a)
            else:
                select_solver_name_list.append(solver_name_b)
        df['select_solver_name'] = select_solver_name_list
        df['select_solver_name_confidence'] = select_solver_name_confidence_list


class EmbeddingMLTreeMultiRegressionSelector(SolverSelector, EmbeddingRowFeatureUtils, TreeUtils, MLRegressionUtils):
    
    def __init__(
        self, 
         accuracy_alpha, time_alpha, dollar_alpha, valid_solver_names, 
         tree_simple_def,
         ml_model_cls, ml_model_kwargs={}, enable_X_norm=True, enable_y_norm=True, embedding_type='mpnet',
    ):
        EmbeddingRowFeatureUtils.__init__(self, embedding_type)
        TreeUtils.__init__(self, valid_solver_names, tree_simple_def)
        MLRegressionUtils.__init__(self, ml_model_cls, ml_model_kwargs, enable_X_norm, enable_y_norm)
        self.valid_solver_names = valid_solver_names
        self.accuracy_alpha = accuracy_alpha
        self.time_alpha = time_alpha
        self.dollar_alpha = dollar_alpha
    
    def apply(self, df):
        self.ensure_text_embedding(df)
        def create_label(row, solver_name_list, metrics_name):
            return np.mean([row["{}_{}".format(solver_name, metrics_name)] for solver_name in solver_name_list])
        model_metrics_name_list = []
        if self.accuracy_alpha != 0:
            model_metrics_name_list.append('accuracy')
        if self.time_alpha != 0:
            model_metrics_name_list.append('time_cost')
        if self.dollar_alpha != 0:
            model_metrics_name_list.append('dollar_cost')
        node_idx_to_model = {}
        for node in self.node_list:
            if node['is_leaf'] is False:
                solver_name_table = node['solver_name_table']
                node_idx = node['node_idx']
                for table_idx, solver_name_list in enumerate(solver_name_table):
                    for metrics_name in model_metrics_name_list:
                        label_name = 'label_node_idx_{}_table_idx_{}_metrics_{}'.format(node_idx, table_idx, metrics_name)
                        func = partial(create_label, solver_name_list=solver_name_list, metrics_name=metrics_name)
                        df[label_name] = df.apply(func, axis=1)
                        print("Build regression model for {}".format(label_name))
                        self.fit_and_transform(df, label_name)
        # select
        def select_solver(row):
            node_idx = 0
            node = self.node_list[0]
            while node['is_leaf'] is False:
                solver_name_table = node['solver_name_table']
                table_idx_info_list = []
                for table_idx, solver_name_list in enumerate(solver_name_table):
                    utility = 0
                    if self.accuracy_alpha != 0:
                        utility += self.accuracy_alpha * row[f"pred_label_node_idx_{node_idx}_table_idx_{table_idx}_metrics_accuracy"]
                    if self.time_alpha != 0:
                        utility += -self.time_alpha * row[f"pred_label_node_idx_{node_idx}_table_idx_{table_idx}_metrics_time_cost"]
                    if self.dollar_alpha != 0:
                        utility += -self.dollar_alpha * row[f"pred_label_node_idx_{node_idx}_table_idx_{table_idx}_metrics_dollar_cost"]
                    table_idx_info_list.append((utility, table_idx))
                table_idx_info_list = sorted(table_idx_info_list, reverse=True)
                select_table_idx = table_idx_info_list[0][1]
                # select next level
                node_idx = node['child_node_idx_list'][select_table_idx]
                node = self.node_list[node_idx]
            return node['solver_name']
        df['select_solver_name'] = df.apply(select_solver, axis=1)
        df['select_solver_name_confidence'] = 1


class EmbeddingMLTreeAccBinaryRegressionSelector(SolverSelector, EmbeddingRowFeatureUtils, TreeUtils, MLRegressionUtils):
    
    def __init__(
        self, 
         accuracy_alpha, time_alpha, dollar_alpha, valid_solver_names, 
         tree_simple_def,
         ml_model_cls, ml_model_kwargs={}, enable_X_norm=True, enable_y_norm=True, embedding_type='mpnet',
    ):
        EmbeddingRowFeatureUtils.__init__(self, embedding_type)
        TreeUtils.__init__(self, valid_solver_names, tree_simple_def)
        MLRegressionUtils.__init__(self, ml_model_cls, ml_model_kwargs, enable_X_norm, enable_y_norm)
        self.valid_solver_names = valid_solver_names
        self.accuracy_alpha = accuracy_alpha
        self.time_alpha = time_alpha
        self.dollar_alpha = dollar_alpha
    
    def apply(self, df):
        self.ensure_text_embedding(df)

        def get_metrics_diff(row, metrics_name, solver_name_list_a, solver_name_list_b):
            group_a_mean = np.mean([row["{}_{}".format(solver_name, metrics_name)] for solver_name in solver_name_list_a])
            group_b_mean = np.mean([row["{}_{}".format(solver_name, metrics_name)] for solver_name in solver_name_list_b])
            return group_a_mean - group_b_mean
        
        def get_estimate_metrics_on_train_set(df, metrics_name, solver_name_list_a, solver_name_list_b):
            # only use train set!
            train_df = df[(df.split == 'TRAIN')]
            func = partial(
                get_metrics_diff, 
                metrics_name=metrics_name, 
                solver_name_list_a=solver_name_list_a, 
                solver_name_list_b=solver_name_list_b,
            )
            diff_series = df.apply(func, axis=1)
            return diff_series.mean()
    
        # select
        def select_solver(row):
            node_idx = 0
            node = self.node_list[0]
            while node['is_leaf'] is False:
                solver_name_table = node['solver_name_table']
                solver_name_list_a = solver_name_table[0]
                solver_name_list_b = solver_name_table[1]

                utility_diff = 0
                utility_diff += self.accuracy_alpha * row[f"pred_node_idx_{node_idx}_accuracy_diff"]
                utility_diff += -self.time_alpha * row[f"pred_node_idx_{node_idx}_time_cost_diff"]
                utility_diff += -self.dollar_alpha * row[f"pred_node_idx_{node_idx}_dollar_cost_diff"]

                # select next level
                if utility_diff > 0:
                    node_idx = node['child_node_idx_list'][0]
                else:
                    node_idx = node['child_node_idx_list'][1]
                node = self.node_list[node_idx]
            return node['solver_name']
            
        node_idx_to_model = {}
        for node in self.node_list:
            if node['is_leaf'] is False:
                solver_name_table = node['solver_name_table']
                assert len(solver_name_table) == 2
                solver_name_list_a = solver_name_table[0]
                solver_name_list_b = solver_name_table[1]
                node_idx = node['node_idx']
                # build acc diff label and regression
                func = partial(
                    get_metrics_diff, metrics_name='accuracy', 
                    solver_name_list_a=solver_name_list_a, solver_name_list_b=solver_name_list_b,
                )
                print(f'Building node_idx_{node_idx}_accuracy_diff ......')
                df[f'node_idx_{node_idx}_accuracy_diff'] = df.apply(func, axis=1)
                self.fit_and_transform(df, f'node_idx_{node_idx}_accuracy_diff')
                # apply diff on train set to test set
                df[f'pred_node_idx_{node_idx}_time_cost_diff'] = get_estimate_metrics_on_train_set(
                    df, 'time_cost', solver_name_list_a, solver_name_list_b
                )
                df[f'pred_node_idx_{node_idx}_dollar_cost_diff'] = get_estimate_metrics_on_train_set(
                    df, 'dollar_cost', solver_name_list_a, solver_name_list_b
                )
        df['select_solver_name'] = df.apply(select_solver, axis=1)
        df['select_solver_name_confidence'] = 1


class EmbeddingMLClassificationSelector(SolverSelector, EmbeddingRowFeatureUtils, MLClassificationUtils):
    
    def __init__(self, accuracy_alpha, time_alpha, dollar_alpha, valid_solver_names, 
        ml_model_cls, ml_model_kwargs={}, enable_X_norm=True, embedding_type='mpnet'):
        EmbeddingRowFeatureUtils.__init__(self, embedding_type)
        MLClassificationUtils.__init__(self, accuracy_alpha, time_alpha, dollar_alpha, valid_solver_names, 
        ml_model_cls, ml_model_kwargs, enable_X_norm)
        
    def apply(self, df):
        self.ensure_text_embedding(df)
        self.fit_and_transform(df)
        df['select_solver_name'] = df['pred_solver_name']


def determine_baseline_solver_name(df, accuracy_alpha, time_alpha, dollar_alpha, valid_solver_names):
    utility_info_list = []
    for solver_name in valid_solver_names:
        train_mean_accuracy = df[df.split == 'TRAIN']['{}_accuracy'.format(solver_name)].mean()
        train_mean_time = df[df.split == 'TRAIN']['{}_time_cost'.format(solver_name)].mean()
        train_mean_dollar = df[df.split == 'TRAIN']['{}_dollar_cost'.format(solver_name)].mean()
        train_mean_utility = train_mean_accuracy * accuracy_alpha - train_mean_time * time_alpha - train_mean_dollar * dollar_alpha
        utility_info_list.append((train_mean_utility, solver_name))
    utility_info_list = sorted(utility_info_list, reverse=True)
    best_solver_name = utility_info_list[0][1]
    return best_solver_name

def calc_row_utility(row, accuracy_alpha, time_alpha, dollar_alpha, solver_name):
    return row[f'{solver_name}_accuracy'] * accuracy_alpha \
        - row[f'{solver_name}_time_cost'] * time_alpha \
        - row[f'{solver_name}_dollar_cost'] * dollar_alpha



class BaselineImprovementSelector(SolverSelector, EmbeddingRowFeatureUtils, MLRegressionUtils):
    
    def __init__(
        self, 
        accuracy_alpha, time_alpha, dollar_alpha, valid_solver_names, 
        ml_model_cls, ml_model_kwargs={}, enable_X_norm=True, enable_y_norm=True, embedding_type='mpnet',
        do_calibration=True
    ):
        EmbeddingRowFeatureUtils.__init__(self, embedding_type)
        MLRegressionUtils.__init__(self, ml_model_cls, ml_model_kwargs, enable_X_norm, enable_y_norm)
        self.valid_solver_names = valid_solver_names
        assert len(self.valid_solver_names) != 1
        self.accuracy_alpha = accuracy_alpha
        self.time_alpha = time_alpha
        self.dollar_alpha = dollar_alpha
        self.do_calibration = do_calibration
        
    def apply(self, df):
        self.ensure_text_embedding(df)
        # step 1, build a diff prediction model for each solver
        for solver_name in self.valid_solver_names:
            func = partial(
                calc_row_utility, 
                accuracy_alpha=self.accuracy_alpha, time_alpha=self.time_alpha, dollar_alpha=self.dollar_alpha, 
                solver_name=solver_name
            )
            df[f'{solver_name}_utility'] = df.apply(func, axis=1)
        baseline_solver_name = determine_baseline_solver_name(df, self.accuracy_alpha, self.time_alpha, self.dollar_alpha, self.valid_solver_names)
#         baseline_solver_name = 'direct'
        self.baseline_solver_name = baseline_solver_name
        print("baseline_solver_name", baseline_solver_name)
        for solver_name in self.valid_solver_names:
            diff_col_name = f'{solver_name}_util_diff_to_baseline'
            df[diff_col_name] = df[f'{solver_name}_utility'] - df[f'{baseline_solver_name}_utility']
            print("Build regression model for {}".format(diff_col_name))
            self.fit_and_transform(df, diff_col_name)
        
        # step 2: calibration (use the dev set)
        if self.do_calibration:
            self.solver_name_to_calibration_model = {}
            eval_df = sample_df[sample_df.split == "DEV"]
            for solver_name in self.valid_solver_names:
                dev_pred_diff_num_list = []
                dev_expected_diff_num_list = []
                for quantile_percent in range(6, 95, 1):
                    # predicted diff num
                    pred_diff_num_lower_bound = eval_df[f'pred_{solver_name}_util_diff_to_baseline'].quantile((quantile_percent - 5) / 100.)
                    pred_diff_num_upper_bound = eval_df[f'pred_{solver_name}_util_diff_to_baseline'].quantile((quantile_percent + 5) / 100.)
                    part_df = eval_df[
                        (eval_df[f'pred_{solver_name}_util_diff_to_baseline'] >= pred_diff_num_lower_bound)
                        &
                        (eval_df[f'pred_{solver_name}_util_diff_to_baseline'] <= pred_diff_num_upper_bound)
                    ]
                    baseline_utility = part_df[f'{self.baseline_solver_name}_utility'].mean()
                    solver_utility = part_df[f'{solver_name}_utility'].mean()
                    expected_diff = solver_utility - baseline_utility
                    dev_pred_diff_num_list.append((pred_diff_num_lower_bound + pred_diff_num_upper_bound) / 2)
                    dev_expected_diff_num_list.append(solver_utility - baseline_utility)
                ir = IsotonicRegression(out_of_bounds='clip', increasing=True)
                ir.fit(dev_pred_diff_num_list, dev_expected_diff_num_list)
                self.solver_name_to_calibration_model[solver_name] = ir
                df[f'calibrated_pred_{solver_name}_util_diff_to_baseline'] = \
                    ir.predict(df[f'pred_{solver_name}_util_diff_to_baseline'].values) 
        else:
            df[f'calibrated_pred_{solver_name}_util_diff_to_baseline'] = df[f'pred_{solver_name}_util_diff_to_baseline']
        
        # step 3 apply
        def apply_selection(row):
            solver_info_list = []
            for solver_name in self.valid_solver_names:
                solver_info_list.append((
                    row[f'calibrated_pred_{solver_name}_util_diff_to_baseline'],
                    solver_name,
                ))
            top_ret = sorted(solver_info_list, reverse=True)[0]
            select_solver_name = top_ret[1]
            select_solver_util_diff = top_ret[0]
            if select_solver_util_diff > 0.00:
                return pd.Series([select_solver_name, select_solver_util_diff], index=['select_solver_name', 'select_solver_util_diff'])
            else:
                return pd.Series([self.baseline_solver_name, select_solver_util_diff], index=['select_solver_name', 'select_solver_util_diff'])
        temp_df = df.apply(apply_selection, axis=1)       
        df['select_solver_name'] = temp_df['select_solver_name']
        df['select_solver_util_diff'] = temp_df['select_solver_util_diff']