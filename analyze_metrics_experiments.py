import pandas as pd
import os
import matplotlib.pyplot as plt
import re
from execute_experiments import Dataset1Configuration


def plot_window_size_grouping_by_logsize(df, selected_column, title, delta=None):
    ############################################################
    # Grouping by logsize
    ############################################################
    df_filtered = df.filter(like=selected_column, axis=1)
    df_filtered.index.name = 'log size'
    if delta:
        df_filtered = df_filtered.filter(like=f'd={delta}', axis=1)

    # maintain only the last number in the column names (window)
    df_plot = df_filtered.rename(
        columns={element: re.sub(r'(\D.*?)(\d+)(?!.*\d)', r'\2', element, count=1)
                 for element in df_filtered.columns.tolist()})

    # sort columns
    ordered_columns = [int(w) for w in df_plot.columns]
    ordered_columns.sort()
    ordered_columns = [str(w) for w in ordered_columns]
    df_plot = df_plot[ordered_columns]

    pattern = '[a-zA-Z]*(\d.*).xes$'
    size = df_plot.index.str.extract(pattern, expand=False)
    df_plot = df_plot.groupby(size).mean().T
    plt.cla()
    plt.clf()
    df_plot.plot(kind='line')
    plt.xlabel('Window size')
    plt.ylabel(selected_column)
    if delta:
        plt.title(f'{title}\nImpact of the window size on the {selected_column} delta={delta}')
    else:
        plt.title(f'{title}\nImpact of the window size on the {selected_column}')
    if 'f_score' in selected_column:
        plt.ylim(0.0, 1.0)
    plt.grid(True)
    # get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    # specify order of items in legend
    order = [1, 2, 3, 0]
    # add legend to plot
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    plt.show()


def plot_window_size_grouping_by_change_pattern(df, selected_column, title, window, delta=None):
    ############################################################
    # Grouping by logsize
    ############################################################
    df_filtered = df.filter(like=selected_column, axis=1)
    df_filtered.index.name = 'change pattern'
    if delta:
        df_filtered = df_filtered.filter(like=f'd={delta}', axis=1)

    # maintain only the last number in the column names (window)
    df_plot = df_filtered.rename(
        columns={element: re.sub(r'(\D.*?)(\d+)(?!.*\d)', r'\2', element, count=1)
                 for element in df_filtered.columns.tolist()})

    # filter only the results for the define window
    df_plot = df_plot.filter(like=f'{window}', axis=1)

    # sort columns
    # ordered_columns = [int(w) for w in df_plot.columns]
    # ordered_columns.sort()
    # ordered_columns = [str(w) for w in ordered_columns]
    # df_plot = df_plot[ordered_columns]

    regexp = '([a-zA-Z]*)\d.*.xes$'
    change_pattern = df_plot.index.str.extract(regexp, expand=False)
    df_plot = df_plot.groupby(change_pattern).mean()
    # sort columns
    # df_plot.sort_index()
    df_plot['labels'] = df_plot.index.str.lower()
    df_plot = df_plot.sort_values('labels').drop('labels', axis=1)
    plt.cla()
    plt.clf()
    df_plot.plot(kind='bar', legend=None)
    plt.xlabel('Change pattern')
    plt.ylabel(selected_column)
    if delta:
        plt.title(f'{title}\n{selected_column} by change pattern - window={window} delta={delta}')
    else:
        plt.title(f'{title}\n{selected_column} by change pattern - window={window}')
    if 'f_score' in selected_column:
        plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.show()


def analyze_metrics_ipdd(input_path, filename, dataset_config, selected_column, title):
    complete_filename = os.path.join(input_path, filename)
    df = pd.read_excel(complete_filename, index_col=0)
    df.index.name = 'logname'
    print(f'Reading file {filename}')
    ############################################################
    # Impact of the window size on the metrics
    ############################################################
    for d in dataset_config.deltas:
        # plot_window_size_grouping_by_logsize(df, selected_column, title, d)
        plot_window_size_grouping_by_change_pattern(df, selected_column, title, 100, d)


def analyze_metrics(input_path, filename, selected_column, title):
    complete_filename = os.path.join(input_path, filename)
    df = pd.read_excel(complete_filename, index_col=0)
    df.index.name = 'logname'
    print(f'Reading file {filename}')
    ############################################################
    # Impact of the window size on the metrics
    ############################################################
    # plot_window_size_grouping_by_logsize(df, selected_column, title)
    plot_window_size_grouping_by_change_pattern(df, selected_column, title, 100)


def dataset1():
    config = Dataset1Configuration()
    f_score_column_ipdd = 'f_score'
    mean_delay_column_ipdd = 'mean_delay'

    ipdd_quality_trace_path = 'C://Users//denisesato//PycharmProjects//IPDD_adaptive_controlflow//data//output' \
                              '//controlflow_adaptive//detection_on_quality_metrics_trace_by_trace//dataset1'
    ipdd_quality_trace_filename = 'metrics_experiments_quality_metrics_trace_by_trace_dataset1.xlsx'
    analyze_metrics_ipdd(ipdd_quality_trace_path, ipdd_quality_trace_filename, config, f_score_column_ipdd,
                         'Quality Metrics - Trace Approach')
    analyze_metrics_ipdd(ipdd_quality_trace_path, ipdd_quality_trace_filename, config, mean_delay_column_ipdd,
                         'Quality Metrics - Trace Approach')

    ipdd_quality_windowing_path = 'C://Users//denisesato//PycharmProjects//IPDD_adaptive_controlflow//data//output' \
                                  '//controlflow_adaptive//detection_on_quality_metrics_fixed_window//dataset1'
    ipdd_quality_windowing_filename = 'metrics_experiments_quality_metrics_fixed_window_dataset1.xlsx'
    analyze_metrics_ipdd(ipdd_quality_windowing_path, ipdd_quality_windowing_filename, config, f_score_column_ipdd,
                         'Quality Metrics - Window Approach')
    analyze_metrics_ipdd(ipdd_quality_windowing_path, ipdd_quality_windowing_filename, config, mean_delay_column_ipdd,
                         'Quality Metrics - Window Approach')

    ipdd_model_similarity_path = 'C://Users//denisesato//PycharmProjects//IPDD_adaptive_controlflow//data//output' \
                                 '//controlflow_adaptive//detection_on_model_similarity_fixed_window//dataset1'
    ipdd_model_similarity_filename = 'metrics_experiments_model_similarity_fixed_window_dataset1.xlsx'
    analyze_metrics_ipdd(ipdd_model_similarity_path, ipdd_model_similarity_filename, config, f_score_column_ipdd,
                         'Model Similarity')
    analyze_metrics_ipdd(ipdd_model_similarity_path, ipdd_model_similarity_filename, config, mean_delay_column_ipdd,
                         'Model Similarity')

    apromore_path = 'C://Users//denisesato//Experimentos_Tese//Apromore//experimento2//dataset1'
    apromore_filename = 'metrics_results_prodrift.xlsx'
    analyze_metrics(apromore_path, apromore_filename, 'f_score awin', 'AWIN')
    analyze_metrics(apromore_path, apromore_filename, 'mean_delay awin', 'AWIN')
    analyze_metrics(apromore_path, apromore_filename, 'f_score fwin', 'FWIN')
    analyze_metrics(apromore_path, apromore_filename, 'mean_delay fwin', 'FWIN')

    # vdd_path = 'C://Users//denisesato//Experimentos_Tese//VDD//dataset1//output_console'
    # vdd_filename = 'metrics_results_vdd.xlsx'
    # analyze_metrics(vdd_path, vdd_filename, 'f_score all', 'ALL')
    # analyze_metrics(vdd_path, vdd_filename, 'mean_delay all', 'ALL')
    # analyze_metrics(vdd_path, vdd_filename, 'f_score cluster', 'CLUSTER')
    # analyze_metrics(vdd_path, vdd_filename, 'mean_delay all', 'CLUSTER')


if __name__ == '__main__':
    dataset1()
