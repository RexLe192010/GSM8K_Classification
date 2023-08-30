import pandas as pd
from functools import partial

def evaluate_row(row, accuracy_alpha, time_alpha, dollar_alpha):
    solver_name = row.select_solver_name
    select_accuracy = row['{}_accuracy'.format(solver_name)]
    select_time_cost = row['{}_time_cost'.format(solver_name)]
    select_dollar_cost = row['{}_dollar_cost'.format(solver_name)]
    select_utility = accuracy_alpha * select_accuracy - time_alpha * select_time_cost - dollar_alpha * select_dollar_cost
    return pd.Series([select_accuracy, select_time_cost, select_dollar_cost, select_utility], index=[
        'select_accuracy', 'select_time_cost', 'select_dollar_cost', 'select_utility',
    ])

def evaluate(df, accuracy_alpha, time_alpha, dollar_alpha, verbose=True):
    func = partial(evaluate_row, accuracy_alpha=accuracy_alpha, time_alpha=time_alpha, dollar_alpha=dollar_alpha)
    temp_df = df.apply(func, axis=1)
    df['select_accuracy'] = temp_df['select_accuracy']
    df['select_time_cost'] = temp_df['select_time_cost']
    df['select_dollar_cost'] = temp_df['select_dollar_cost']
    df['select_utility'] = temp_df['select_utility']
    metrics = {}
    for split in ['TRAIN', 'DEV', 'TEST']:
        mean_acc = df[df.split == split]['select_accuracy'].mean()
        mean_time_cost = df[df.split == split]['select_time_cost'].mean()
        mean_dollar_cost = df[df.split == split]['select_dollar_cost'].mean()
        mean_utility = df[df.split == split]['select_utility'].mean()
        if verbose:
            print("Split: {:>8}  Util: {:<8.4f} Acc: {:>8.3f}%  Time (sec): {:<8.2f} Dollar (1000Q): {:<8.3f}".format(
                split,
                mean_utility,
                mean_acc * 100,
                mean_time_cost,
                mean_dollar_cost * 1000,
            ))
        metrics['{}_{}'.format(split.lower(), 'mean_acc')] = mean_acc
        metrics['{}_{}'.format(split.lower(), 'mean_time_cost')] = mean_time_cost
        metrics['{}_{}'.format(split.lower(), 'mean_dollar_cost')] = mean_dollar_cost
        metrics['{}_{}'.format(split.lower(), 'mean_utility')] = mean_utility
    return metrics