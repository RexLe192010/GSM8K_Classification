from src.utils import ensure_text_embedding, ensure_openai_text_embedding
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import numpy as np

class EmbeddingRowFeatureUtils(object):

    def __init__(self, embedding_type):
        self.embedding_type = embedding_type

    def ensure_text_embedding(self, df):
        if self.embedding_type == 'mpnet':
            ensure_text_embedding(df)
            self.embedding_col_name = 'text_embedding'
        elif self.embedding_type == 'openai':
            ensure_openai_text_embedding(df)
            self.embedding_col_name = 'openai_text_embedding'

    def get_row_feature(self, row):
        return row[self.embedding_col_name]

class MLRegressionUtils(object):
    
    def __init__(self, ml_model_cls, ml_model_kwargs={}, enable_X_norm=True, enable_y_norm=True):
        self.ml_model_cls = ml_model_cls
        self.ml_model_kwargs = ml_model_kwargs
        self.enable_X_norm = enable_X_norm
        self.enable_y_norm = enable_y_norm

    def get_row_feature(self, row):
        raise NotImplementedError

    def fit_and_transform(self, df, target_label):
        # train samples
        X_train = []
        y_train = []
        train_df = df[df.split == 'TRAIN']
        for _, row in train_df.iterrows():
            X_train.append(self.get_row_feature(row))
            y_train.append([row[target_label]])
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        if self.enable_X_norm:
            X_scaler = StandardScaler().fit(X_train)
            X_train = X_scaler.transform(X_train)
            self.X_scaler = X_scaler
        if self.enable_y_norm:
            y_scaler = StandardScaler().fit(y_train)
            y_train = y_scaler.transform(y_train)
            self.y_scaler = y_scaler
        # all_samples
        X = []
        for _, row in df.iterrows():
            X.append(self.get_row_feature(row))
        X = np.array(X)
        if self.enable_X_norm:
             X = X_scaler.transform(X)
        # train the model
        self.model = self.ml_model_cls(**self.ml_model_kwargs)
        self.model.fit(X_train, y_train)
        pred_y = self.model.predict(X).reshape(-1, 1)
        if self.enable_X_norm:
            pred_y = y_scaler.inverse_transform(pred_y)
        df['pred_' + target_label] = pred_y

class TreeUtils(object):
    
    def __init__(self, valid_solver_names, tree_simple_def):
        self.node_list = self.create_node_list(tree_simple_def)
        node_tree_solver_names = [node['solver_name'] for node in self.node_list if node['is_leaf'] is True]
        assert sorted(valid_solver_names) == sorted(node_tree_solver_names)

    @staticmethod
    def create_node_list(tree_simple_def):
        root_node = {'parent_idx': 0, 'node_idx': 0, 'is_leaf': False, 'child_node_idx_list': []}
        node_list = [root_node]
        node_idx = 1
        node_idx_to_solver_name_info = defaultdict(lambda : defaultdict(list))
        def create_node(parent_idx, list_def):
            nonlocal node_idx
            for item in list_def:
                if isinstance(item, str):
                    node = {
                        'parent_idx': parent_idx,
                        'node_idx': node_idx,
                        'is_leaf': True,
                        'solver_name': item,
                    }
                    node_list[parent_idx]['child_node_idx_list'].append(node_idx)
                    node_list.append(node)
                    node_idx += 1
                elif isinstance(item, list):
                    cur_node_idx = node_idx
                    node = {
                        'parent_idx': parent_idx,
                        'node_idx': node_idx,
                        'is_leaf': False,
                        'child_node_idx_list': [],
                    }
                    node_list[parent_idx]['child_node_idx_list'].append(cur_node_idx)
                    node_list.append(node)
                    node_idx += 1
                    create_node(cur_node_idx, item)

        def ensure_list_of_str(x):
            new_x = []
            for item in x:
                if isinstance(item, str):
                    new_x.append(item)
                elif isinstance(item, list):
                    new_x.extend(item)
            return new_x

        def gather_solver_name(node_idx):
            node = node_list[node_idx]
            if node['is_leaf']:
                return [node['solver_name']]
            else:
                node_solver_name_table = []
                for child_node_idx in node['child_node_idx_list']:
                    child_node_solver_name_table = gather_solver_name(child_node_idx)
                    node_solver_name_table.append(ensure_list_of_str(child_node_solver_name_table))
                node['solver_name_table'] = node_solver_name_table
                return node_solver_name_table
        create_node(0, tree_simple_def)
        gather_solver_name(0)
        return node_list


class MLClassificationUtils(object):
    
    def __init__(
        self, accuracy_alpha, time_alpha, dollar_alpha, valid_solver_names, 
        ml_model_cls, ml_model_kwargs={}, enable_X_norm=True
    ):
        self.accuracy_alpha = accuracy_alpha
        self.time_alpha = time_alpha
        self.dollar_alpha = dollar_alpha
        self.valid_solver_names = valid_solver_names
        self.solver_name_to_class_id = {solver_name: idx for idx, solver_name in enumerate(valid_solver_names)}
        self.class_id_to_solver_name = {idx: solver_name for idx, solver_name in enumerate(valid_solver_names)}
        self.ml_model_cls = ml_model_cls
        self.ml_model_kwargs = ml_model_kwargs
        self.enable_X_norm = enable_X_norm

    def get_row_feature(self, row):
        raise NotImplementedError

    def fit_and_transform(self, df):
        # train samples
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
            if solver_name_info_list[0][0] - solver_name_info_list[1][0] > 0:
                return select_solver_name
            else:
                return None
        
        # this oracle!
        df['oracle_solver_name'] = df.apply(get_oracle_solver_name, axis=1)
        df['oracle_class_id'] = df['oracle_solver_name'].map(
            lambda solver_name: self.solver_name_to_class_id[solver_name] if solver_name is not None else None
        )
        # train
        train_df = df[(df.split == 'TRAIN') & (df.oracle_solver_name.notnull())].copy()
        print("Train samples: {}".format(len(train_df)))
        X_train = []
        y_train = []
        for _, row in train_df.iterrows():
            X_train.append(self.get_row_feature(row))
            y_train.append(row['oracle_class_id'])
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        if self.enable_X_norm:
            X_scaler = StandardScaler().fit(X_train)
            X_train = X_scaler.transform(X_train)
            self.X_scaler = X_scaler
        self.X_train = X_train
        self.y_train = y_train

        # all_samples
        X = []
        for _, row in df.iterrows():
            X.append(self.get_row_feature(row))
        X = np.array(X)
        if self.enable_X_norm:
             X = X_scaler.transform(X)
        # train the model
        self.model = self.ml_model_cls(**self.ml_model_kwargs)
        self.model.fit(X_train, y_train)
        df['pred_y_class_id'] = self.model.predict(X)
        df['pred_solver_name'] = df['pred_y_class_id'].map(lambda cls_id: self.class_id_to_solver_name[cls_id])