import pandas as pd
import os
import matplotlib.pyplot as plt
import re
from execute_experiments import Dataset1Configuration


def plot_window_size(df, delta, title):
    ############################################################
    # Grouping by logsize
    ############################################################
    df_fscore = df.filter(like='f_score', axis=1)
    df_fscore.index.name = 'log size'
    df_fscore_delta = df_fscore.filter(like=f'd={delta}', axis=1)
    df_plot = df_fscore_delta.rename(columns={element: re.sub(r'(^.*?(?=sp=)sp=)', '', element)
                                              for element in df_fscore_delta.columns.tolist()})

    pattern = '[a-zA-Z]*(\d.*).xes$'
    s = df_plot.index.str.extract(pattern, expand=False)
    df_plot = df_plot.groupby(s).mean().T
    plt.cla()
    plt.clf()
    plt.figure()
    df_plot.plot(kind='line')
    plt.xlabel('Window size')
    plt.ylabel('F-score')
    plt.title(f'{title}\nImpact of the window size on the F-score delta={delta}')
    plt.ylim(0.0, 1.0)
    plt.grid(True)
    # get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()
    # specify order of items in legend
    order = [1, 2, 3, 0]
    # add legend to plot
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
    plt.show()


def analyze_metrics(input_path, filename, dataset_config, title):
    complete_filename = os.path.join(input_path, filename)
    df = pd.read_excel(complete_filename, index_col=0)
    df.index.name = 'logname'
    print(f'Reading file {filename}')
    ############################################################
    # Impact of the window size on the metrics
    ############################################################
    for d in dataset_config.deltas:
        plot_window_size(df, d, title)


def dataset1():
    config = Dataset1Configuration()
    ipdd_quality_trace_path = 'C://Users//denisesato//PycharmProjects//IPDD_adaptive_controlflow//data//output' \
                              '//controlflow_adaptive//detection_on_quality_metrics_trace_by_trace//dataset1'
    ipdd_quality_trace_filename = 'metrics_experiments_quality_trace_by_trace_dataset1.xlsx'
    analyze_metrics(ipdd_quality_trace_path, ipdd_quality_trace_filename, config, 'Quality Metrics - Trace Approach')

    ipdd_quality_windowing_path = 'C://Users//denisesato//PycharmProjects//IPDD_adaptive_controlflow//data//output' \
                                  '//controlflow_adaptive//detection_on_quality_metrics_fixed_window//dataset1'
    ipdd_quality_windowing_filename = 'metrics_experiments_quality_fixed_window_dataset1.xlsx'
    analyze_metrics(ipdd_quality_windowing_path, ipdd_quality_windowing_filename, config,
                    'Quality Metrics - Window Approach')

    ipdd_model_similarity_path = 'C://Users//denisesato//PycharmProjects//IPDD_adaptive_controlflow//data//output' \
                                 '//controlflow_adaptive//detection_on_model_similarity_fixed_window//dataset1'
    ipdd_model_similarity_filename = 'metrics_experiments_model_similarity_fixed_window_dataset1.xlsx'
    analyze_metrics(ipdd_model_similarity_path, ipdd_model_similarity_filename, config, 'Model Similarity')


if __name__ == '__main__':
    dataset1()
