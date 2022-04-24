from calculate_evaluation_metrics import calculate_metrics_dataset1


def define_change_points_dataset1(inter_drift_distance):
    actual_change_points = []
    for i in range(inter_drift_distance, inter_drift_distance * 10, inter_drift_distance):
        actual_change_points.append(i)
    return actual_change_points


def dataset1():
    # information about the dataset
    scenarios = [
        '2.5k',
        '5k',
        '7.5k',
        '10k',
    ]

    actual_change_points = {
        '2.5k': define_change_points_dataset1(250),
        '5k': define_change_points_dataset1(500),
        '7.5k': define_change_points_dataset1(750),
        '10k': define_change_points_dataset1(1000)
    }

    error_tolerance = {
        '2.5k': 250,
        '5k': 500,
        '7.5k': 750,
        '10k': 1000
    }

    number_of_instances = {
        '2.5k': 2500,
        '5k': 5000,
        '7.5k': 7500,
        '10k': 10000
    }

    metrics = [
        'f_score',
        'mean_delay',
        'FPR'
    ]

    ipdd_quality_trace_path = 'C://Users//denisesato//PycharmProjects//IPDD_adaptive_controlflow//data//output//controlflow_adaptive//detection_on_quality_metrics_trace_by_trace//dataset1'
    ipdd_quality_trace_filename = 'experiments_quality_trace_by_trace_dataset1.xlsx'
    calculate_metrics_dataset1(ipdd_quality_trace_path, ipdd_quality_trace_filename, metrics, scenarios,
                               actual_change_points, number_of_instances,
                               error_tolerance, save_input_for_calculation=True)

    # prodrift_filepath = 'C://Users//denisesato//OneDrive//Documents//Doutorado//Tese//experiments//Apromore//dataset1'
    # prodrift_filename = 'results_prodrift.xlsx'
    # calculate_metrics_dataset1(prodrift_filepath, prodrift_filename, metrics, scenarios, actual_change_points,
    #                            number_of_instances, error_tolerance, save_input_for_calculation=True)


if __name__ == '__main__':
    dataset1()
