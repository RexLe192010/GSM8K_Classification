from src.utils import ensure_text_embedding, ensure_openai_text_embedding
from sentence_transformers import util as sbert_util
from tqdm import tqdm
from collections import defaultdict
from src.selector.base import SolverSelector
import numpy as np

class EmbeddingBasedRetrivalSelector(SolverSelector):
    
    def __init__(self, accuracy_alpha, time_alpha, dollar_alpha, valid_solver_names, top_k, embedding_type='mpnet'):
        self.accuracy_alpha = accuracy_alpha
        self.time_alpha = time_alpha
        self.dollar_alpha = dollar_alpha
        self.valid_solver_names = valid_solver_names
        self.top_k = top_k
        self.embedding_type = embedding_type

    def apply(self, df):
        if self.embedding_type == 'mpnet':
            ensure_text_embedding(df)
            embedding_col_name = 'text_embedding'
        elif self.embedding_type == 'openai':
            ensure_openai_text_embedding(df)
            embedding_col_name = 'openai_text_embedding'
        # only use train df to build experience pool
        train_df = df[df.split == 'TRAIN'].copy()
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
        train_df['oracle_solver_name'] = df.apply(get_oracle_solver_name, axis=1)
        train_set_embeddings = []
        train_set_oracle_solver_names = []
        for _, row in train_df.iterrows():
            train_set_embeddings.append(row[embedding_col_name])
            train_set_oracle_solver_names.append(row.oracle_solver_name)
        train_set_embeddings = np.array(train_set_embeddings)

        # apply
        embeddings = []
        for _, row in df.iterrows():
            embeddings.append(row[embedding_col_name])
        embeddings = np.array(embeddings)
        score_table = sbert_util.cos_sim(embeddings, train_set_embeddings)
        select_solver_name_list = []
        for i in tqdm(range(embeddings.shape[0])):
            scores = score_table[i, :]
            top_indices = [int(index) for index in np.argsort(scores)[-self.top_k:]][::-1]
            solver_counter = defaultdict(int)
            for idx in top_indices:
                solver_counter[train_set_oracle_solver_names[idx]] += 1
            solver_info_list = [(v, k) for k, v in solver_counter.items()]
            solver_info_list = sorted(solver_info_list, reverse=True)
            select_solver_name_list.append(solver_info_list[0][1])
        df['select_solver_name'] = select_solver_name_list
        df['select_solver_name_confidence'] = 1
