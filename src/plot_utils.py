import plotly.express as px
import plotly.graph_objects as go
from src.evaluate import evaluate



def plot_against_cost_alpha_list(
    sample_df,
    solver_config_list,
    update_layout_kwargs={},
    accuracy_alpha = 10, 
    time_alpha = 0., 
    dollar_alpha_list = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],

):
    fig = go.Figure()
    for solver_config in solver_config_list:
        print("Add plot for {}".format(solver_config['name']))
        if solver_config['is_static']:
            selecter = solver_config['selecter']()
            selecter.apply(sample_df)
            evaluate_ret = evaluate(sample_df, accuracy_alpha, time_alpha, dollar_alpha_list[0], verbose=False)
            dollar_cost_1k = evaluate_ret['test_mean_dollar_cost'] * 1000
            accuracy = evaluate_ret['test_mean_acc']
            fig.add_trace(
                go.Scatter(x=[dollar_cost_1k], y=[accuracy],
                    mode='lines+markers',
                    marker_size=12,
                    name=solver_config['name'],
                    marker_symbol=solver_config['marker_symbol'],
                )
            )
        else:
            dollar_cost_1k_list = []
            accuracy_list = []
            for dollar_alpha in dollar_alpha_list:
                selecter = solver_config['selecter'](accuracy_alpha=accuracy_alpha, time_alpha=time_alpha, dollar_alpha=dollar_alpha)
                selecter.apply(sample_df)
                evaluate_ret = evaluate(sample_df, accuracy_alpha, time_alpha, dollar_alpha, verbose=False)
                dollar_cost_1k_list.append(evaluate_ret['test_mean_dollar_cost'] * 1000)
                accuracy_list.append(evaluate_ret['test_mean_acc'])
            fig.add_trace(
                go.Scatter(x=dollar_cost_1k_list, y=accuracy_list,
                    mode='lines+markers',
                    marker_size=12,
                    name=solver_config['name'],
                    marker_symbol=solver_config['marker_symbol'],
                )
            )
    kwargs = dict(
        title="Accuracy v.s. dollar",
        width=600,
        height=400,
        xaxis_title="Average dollar (1K Query)",
        yaxis_title="Average Accuracy",
    )
    kwargs.update(update_layout_kwargs)
    fig.update_layout(**kwargs)
    fig.show()
    return fig


def plot_utility_against_cost_alpha_list(
    sample_df,
    solver_config_list,
    update_layout_kwargs={},
    accuracy_alpha = 10, 
    time_alpha = 0., 
    dollar_alpha_list = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],

):
    fig = go.Figure()
    for solver_config in solver_config_list:
        print("Add plot for {}".format(solver_config['name']))
        if solver_config['is_static']:
            selecter = solver_config['selecter']()
            selecter.apply(sample_df)
            utility_list = []
            for dollar_alpha in dollar_alpha_list:
                evaluate_ret = evaluate(sample_df, accuracy_alpha, time_alpha, dollar_alpha, verbose=False)
                utility = evaluate_ret['test_mean_utility']
                utility_list.append(utility)
            fig.add_trace(
                go.Scatter(x=dollar_alpha_list, y=utility_list,
                    mode='lines+markers',
                    marker_size=12,
                    name=solver_config['name'],
                    marker_symbol=solver_config['marker_symbol'],
                )
            )
        else:
            utility_list = []
            for dollar_alpha in dollar_alpha_list:
                selecter = solver_config['selecter'](accuracy_alpha=accuracy_alpha, time_alpha=time_alpha, dollar_alpha=dollar_alpha)
                selecter.apply(sample_df)
                evaluate_ret = evaluate(sample_df, accuracy_alpha, time_alpha, dollar_alpha, verbose=False)
                utility = evaluate_ret['test_mean_utility']
                utility_list.append(utility)
            fig.add_trace(
                go.Scatter(x=dollar_alpha_list, y=utility_list,
                    mode='lines+markers',
                    marker_size=12,
                    name=solver_config['name'],
                    marker_symbol=solver_config['marker_symbol'],
                )
            )
    kwargs = dict(
        title="Utility v.s. dollar_alpha",
        width=600,
        height=400,
        xaxis_title="dollar_alpha",
        yaxis_title="Utility (Test Split)",
    )
    kwargs.update(update_layout_kwargs)
    fig.update_layout(**kwargs)
    fig.show()
    return fig