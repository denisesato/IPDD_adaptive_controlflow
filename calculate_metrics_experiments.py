from evaluation_metrics import calculate_metrics_dataset1


def define_change_points_dataset1(inter_drift_distance):
    actual_change_points = []
    for i in range(inter_drift_distance, inter_drift_distance * 10, inter_drift_distance):
        actual_change_points.append(i)
    return actual_change_points


# defined metrics
metrics = [
    'f_score',
    'mean_delay',
    'mean_detection_delay',
    'FPR'
]

# information about the dataset
scenarios_ds1 = [
    '2.5k',
    '5k',
    '7.5k',
    '10k',
]

actual_change_points_ds1 = {
    '2.5k': define_change_points_dataset1(250),
    '5k': define_change_points_dataset1(500),
    '7.5k': define_change_points_dataset1(750),
    '10k': define_change_points_dataset1(1000)
}

# for files that do not follow the correct pattern
exceptions_in_actual_change_points_ds1 = {
    'cb10k.xes':
        {'actual_change_points': [5000],
         'number_of_instances': 5000},
    'lp2.5k.xes':
        {'actual_change_points': define_change_points_dataset1(500),
         'number_of_instances': 5000},
    'lp5k.xes':
        {'actual_change_points': define_change_points_dataset1(1000),
         'number_of_instances': 1000},
    'lp7.5k.xes':
        {'actual_change_points': [1000, 3500, 4000, 6500, 7000, 9500, 10000, 12500, 13000],
         'number_of_instances': 15000},
    'lp10k.xes':
        {'actual_change_points': [1000, 3500, 4000, 6500, 7000, 9500, 10000, 12500, 13000],
         'number_of_instances': 15000},
    're2.5k.xes':
        {'actual_change_points': define_change_points_dataset1(500),
         'number_of_instances': 5000},
    're5k.xes':
        {'actual_change_points': define_change_points_dataset1(1000),
         'number_of_instances': 10000},
    're7.5k.xes':
        {'actual_change_points': [1000, 2000, 2500, 3500, 4000, 5000, 5500, 6500, 7000, 8000, 8500, 9500, 10000,
                                  11000, 11500,
                                  12500, 13000],
         'number_of_instances': 15000},
    're10k.xes':
        {'actual_change_points': define_change_points_dataset1(2000),
         'number_of_instances': 20000},
}

number_of_instances_ds1 = {
    '2.5k': 2500,
    '5k': 5000,
    '7.5k': 7500,
    '10k': 10000
}


def dataset1():
    ipdd_quality_trace_path = 'data//output//controlflow_adaptive//detection_on_quality_metrics_trace_by_trace' \
                              '//dataset1'
    ipdd_quality_trace_filename = 'experiments_quality_trace_by_trace_dataset1.xlsx'

    calculate_metrics_dataset1(ipdd_quality_trace_path, ipdd_quality_trace_filename, metrics, scenarios_ds1,
                               actual_change_points_ds1, exceptions_in_actual_change_points_ds1,
                               number_of_instances_ds1,
                               save_input_for_calculation=True)

    ipdd_quality_windowing_path = 'data//output//controlflow_adaptive//detection_on_quality_metrics_fixed_window' \
                                  '//dataset1'
    ipdd_quality_windowing_filename = 'experiments_quality_fixed_window_dataset1.xlsx'
    calculate_metrics_dataset1(ipdd_quality_windowing_path, ipdd_quality_windowing_filename, metrics, scenarios_ds1,
                               actual_change_points_ds1, exceptions_in_actual_change_points_ds1,
                               number_of_instances_ds1,
                               save_input_for_calculation=True)

    ipdd_model_similarity_path = 'data//output//controlflow_adaptive//detection_on_model_similarity_fixed_window' \
                                 '//dataset1'
    ipdd_model_similarity_filename = 'experiments_model_similarity_fixed_window_dataset1.xlsx'
    calculate_metrics_dataset1(ipdd_model_similarity_path, ipdd_model_similarity_filename, metrics, scenarios_ds1,
                               actual_change_points_ds1, exceptions_in_actual_change_points_ds1,
                               number_of_instances_ds1,
                               save_input_for_calculation=True)

    prodrift_filepath = 'C://Users//denisesato//Experimentos_Tese//Apromore//dataset1'
    prodrift_filename = 'results_prodrift.xlsx'
    calculate_metrics_dataset1(prodrift_filepath, prodrift_filename, metrics, scenarios_ds1, actual_change_points_ds1,
                               exceptions_in_actual_change_points_ds1, number_of_instances_ds1,
                               save_input_for_calculation=True)


if __name__ == '__main__':
    dataset1()
