import pandas as pd
import os
from calculate_metrics_experiments import metrics


def analyze_metrics(input_path, filename):
    complete_filename = os.path.join(input_path, filename)
    df = pd.read_excel(complete_filename, index_col=0)
    print(f'Reading file {filename}')



def dataset1():
    ipdd_quality_trace_path = 'C://Users//denisesato//PycharmProjects//IPDD_adaptive_controlflow//data//output' \
                              '//controlflow_adaptive//detection_on_quality_metrics_trace_by_trace//dataset1'
    ipdd_quality_trace_filename = 'metrics_experiments_quality_trace_by_trace_dataset1.xlsx'
    analyze_metrics(ipdd_quality_trace_path, ipdd_quality_trace_filename)

    ipdd_quality_windowing_path = 'C://Users//denisesato//PycharmProjects//IPDD_adaptive_controlflow//data//output' \
                                  '//controlflow_adaptive//detection_on_quality_metrics_fixed_window//dataset1'
    ipdd_quality_windowing_filename = 'metrics_experiments_quality_fixed_window_dataset1.xlsx'

    ipdd_model_similarity_path = 'C://Users//denisesato//PycharmProjects//IPDD_adaptive_controlflow//data//output' \
                                 '//controlflow_adaptive//detection_on_quality_metrics_fixed_window//dataset1'
    ipdd_model_similarity_filename = 'metrics_experiments_quality_fixed_window_dataset1.xlsx'


if __name__ == '__main__':
    dataset1()
