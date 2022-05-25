import pandas as pd
import os
import matplotlib.pyplot as plt
import re
from execute_experiments import Dataset1Configuration, Dataset2Configuration

metric_key = 'metric'
path_key = 'path'
filename_key = 'filename'
series_key = 'series'
delta_key = 'delta'


def plot_window_size_grouping_by_logsize(df, selected_column, title, order=None, delta=None):
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
    if order:
        # get handles and labels
        handles, labels = plt.gca().get_legend_handles_labels()
        # specify order of items in legend
        # add legend to plot
        plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    else:
        plt.legend()
    plt.show()


def plot_window_size_grouping_by_change_pattern(df, selected_column, title, window=None, delta=None):
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

    # filter only the results for the defined window if the parameter is set
    if window:
        df_plot = df_plot.filter(like=f'{window}', axis=1)
        window_str = f'window {window}'

    # grouped by change pattern
    regexp = '([a-zA-Z]*)\d.*.xes$'
    change_pattern = df_plot.index.str.extract(regexp, expand=False)
    df_plot = df_plot.groupby(change_pattern).mean()

    # if window is not informed calculate the mean value considering all windows
    if not window:
        df_plot = df_plot.T.mean().to_frame()
        window_str = f'all windows'

    # sort columns
    df_plot['labels'] = df_plot.index.str.lower()
    df_plot = df_plot.sort_values('labels').drop('labels', axis=1)
    plt.cla()
    plt.clf()
    df_plot.plot(kind='bar', legend=None)
    plt.xlabel('Change pattern')
    plt.ylabel(selected_column)
    if delta:
        plt.title(f'{title}\n{selected_column} by change pattern - {window_str} - delta={delta}')
    else:
        plt.title(f'{title}\n{selected_column} by change pattern - {window_str}')
    if 'f_score' in selected_column:
        plt.ylim(0.0, 1.0)
    if 'mean_delay' in selected_column:
        plt.ylim(0, 300)
    plt.tight_layout()
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
    order = None
    if dataset_config.order_legend:
        order = dataset_config.order_legend
    for d in dataset_config.deltas:
        plot_window_size_grouping_by_logsize(df, selected_column, title, order, d)
        plot_window_size_grouping_by_change_pattern(df, selected_column, title, delta=d)
        plot_window_size_grouping_by_change_pattern(df, selected_column, title, window=100, delta=d)


def analyze_metrics(input_path, filename, selected_column, title):
    complete_filename = os.path.join(input_path, filename)
    df = pd.read_excel(complete_filename, index_col=0)
    df.index.name = 'logname'
    print(f'Reading file {filename}')
    ############################################################
    # Impact of the window size on the metrics
    ############################################################
    order = None
    if dataset_config.order_legend:
        order = dataset_config.order_legend
    plot_window_size_grouping_by_logsize(df, selected_column, title, order)
    plot_window_size_grouping_by_change_pattern(df, selected_column, title)
    plot_window_size_grouping_by_change_pattern(df, selected_column, title, window=100)


def analyze_dataset(dataset_config, dataset_name):
    f_score_column_ipdd = 'f_score'
    mean_delay_column_ipdd = 'mean_delay'

    # ipdd_quality_trace_path = f'C://Users//denisesato//PycharmProjects//IPDD_adaptive_controlflow//data//output' \
    #                           f'//controlflow_adaptive//detection_on_quality_metrics_trace_by_trace//{dataset_name}'
    # ipdd_quality_trace_filename = 'metrics_experiments_quality_metrics_trace_by_trace.xlsx'
    # analyze_metrics_ipdd(ipdd_quality_trace_path, ipdd_quality_trace_filename, dataset_config, f_score_column_ipdd,
    #                      'Quality Metrics - Trace Approach')
    # analyze_metrics_ipdd(ipdd_quality_trace_path, ipdd_quality_trace_filename, dataset_config, mean_delay_column_ipdd,
    #                      'Quality Metrics - Trace Approach')

    # ipdd_quality_windowing_path = f'C://Users//denisesato//PycharmProjects//IPDD_adaptive_controlflow//data//output' \
    #                               f'//controlflow_adaptive//detection_on_quality_metrics_fixed_window//{dataset_name}'
    # ipdd_quality_windowing_filename = 'metrics_experiments_quality_metrics_fixed_window.xlsx'
    # analyze_metrics_ipdd(ipdd_quality_windowing_path, ipdd_quality_windowing_filename, dataset_config, f_score_column_ipdd,
    #                      'Quality Metrics - Window Approach')
    # analyze_metrics_ipdd(ipdd_quality_windowing_path, ipdd_quality_windowing_filename, dataset_config, mean_delay_column_ipdd,
    #                      'Quality Metrics - Window Approach')

    # ipdd_model_similarity_path = f'C://Users//denisesato//PycharmProjects//IPDD_adaptive_controlflow//data//output' \
    #                              f'//controlflow_adaptive//detection_on_model_similarity_fixed_window//{dataset_name}'
    # ipdd_model_similarity_filename = 'metrics_experiments_model_similarity_fixed_window.xlsx'
    # analyze_metrics_ipdd(ipdd_model_similarity_path, ipdd_model_similarity_filename, dataset_config, f_score_column_ipdd,
    #                      'Model Similarity')
    # analyze_metrics_ipdd(ipdd_model_similarity_path, ipdd_model_similarity_filename, dataset_config, mean_delay_column_ipdd,
    #                      'Model Similarity')

    apromore_path = f'C://Users//denisesato//Experimentos_Tese//Apromore//{dataset_name}'
    apromore_filename = 'metrics_results_prodrift.xlsx'
    analyze_metrics(apromore_path, apromore_filename, 'f_score awin', 'AWIN')
    analyze_metrics(apromore_path, apromore_filename, 'mean_delay awin', 'AWIN')
    analyze_metrics(apromore_path, apromore_filename, 'f_score fwin', 'FWIN')
    analyze_metrics(apromore_path, apromore_filename, 'mean_delay fwin', 'FWIN')

    # vdd_path = f'C://Users//denisesato//Experimentos_Tese//VDD//{dataset_name}//output_console'
    # vdd_filename = 'metrics_results_vdd.xlsx'
    # analyze_metrics(vdd_path, vdd_filename, 'f_score all', 'ALL')
    # analyze_metrics(vdd_path, vdd_filename, 'mean_delay all', 'ALL')
    # analyze_metrics(vdd_path, vdd_filename, 'f_score cluster', 'CLUSTER')
    # analyze_metrics(vdd_path, vdd_filename, 'mean_delay all', 'CLUSTER')


def generate_plot_tools(approaches, metric_name):
    # firstly enrich dict with dataframe from excel
    for key in approaches.keys():
        input_path = approaches[key][path_key]
        filename = approaches[key][filename_key]
        complete_filename = os.path.join(input_path, filename)
        df = pd.read_excel(complete_filename, index_col=0)
        df.index.name = 'logname'
        print(f'Reading file {filename}')
        # filter the selected metric
        df_filtered = df.filter(like=approaches[key][metric_key], axis=1)
        df_filtered.index.name = 'log size'
        # if IPDD filter the selected delta
        if 'IPDD' in key:
            df_filtered = df_filtered.filter(like=f'd={approaches[key][delta_key]}', axis=1)
        # maintain only the last number in the column names (window)
        df_filtered = df_filtered.rename(
            columns={element: re.sub(r'(\D.*?)(\d+)(?!.*\d)', r'\2', element, count=1)
                     for element in df_filtered.columns.tolist()})
        series = df_filtered.mean()
        series.name = key
        approaches[key][series_key] = series

    # combine all approaches into one dataframe
    df_plot = pd.concat([approaches[approach][series_key] for approach in approaches.keys()], axis=1)
    df_plot.sort_index(axis=1, inplace=True)
    plt.cla()
    plt.clf()
    df_plot.plot(kind='line')
    plt.xlabel('Window size')
    plt.ylabel(metric_name)
    plt.title(f'Impact of the window size on the {metric_name}')
    if 'f_score' in metric_name:
        plt.ylim(0.0, 1.0)
    plt.grid(True)
    plt.legend()
    plt.show()


def compare_tools_dataset(dataset_config, dataset_name, metric_name):
    approaches = {
        'IPDD Quality - Trace':
            {
                metric_key: metric_name,
                path_key: f'C://Users//denisesato//PycharmProjects//IPDD_adaptive_controlflow//data//output' \
                        f'//controlflow_adaptive//detection_on_quality_metrics_trace_by_trace//{dataset_name}',
                filename_key: 'metrics_experiments_quality_metrics_trace_by_trace.xlsx',
                delta_key: 0.002
            },
        'Apromore ProDrift - AWIN':
            {
                metric_key: f'{metric_name} awin',
                path_key: f'C://Users//denisesato//Experimentos_Tese//Apromore//{dataset_name}',
                filename_key: 'metrics_results_prodrift.xlsx'
            },
        'Apromore ProDrift - FWIN':
            {
                metric_key: f'{metric_name} fwin',
                path_key: f'C://Users//denisesato//Experimentos_Tese//Apromore//{dataset_name}',
                filename_key: 'metrics_results_prodrift.xlsx'
            },
    }
    generate_plot_tools(approaches, metric_name)


if __name__ == '__main__':
    # dataset_config = Dataset1Configuration()
    # analyze_dataset(dataset_config, "dataset1")
    dataset_config = Dataset2Configuration()
    # analyze_dataset(dataset_config, "dataset2")

    compare_tools_dataset(dataset_config, "dataset2", 'f_score')
    compare_tools_dataset(dataset_config, "dataset2", 'mean_delay')
