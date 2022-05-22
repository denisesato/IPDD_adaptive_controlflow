import os
import pandas as pd
from evaluation_metrics import change_points_key
from detect_controlflow_drift import apply_detector_on_quality_metrics_trace_by_trace, QualityDimension, \
    SimilarityMetric, apply_detector_on_model_similarity_fixed_window, apply_detector_on_quality_metrics_fixed_window, \
    apply_detector_on_quality_metrics_adaptive_window


def define_change_points_dataset1(inter_drift_distance):
    actual_change_points = []
    for i in range(inter_drift_distance, inter_drift_distance * 10, inter_drift_distance):
        actual_change_points.append(i)
    return actual_change_points


class Dataset1Configuration:
    ###############################################################
    # Information about the data for performing the experiments
    ###############################################################
    input_folder = 'data/input/logs/Controlflow/dataset1'
    lognames2500 = [
        'cb2.5k.xes',
        'cd2.5k.xes',
        'cf2.5k.xes',
        'cm2.5k.xes',
        'cp2.5k.xes',
        # 'fr2.5k.xes',
        'IOR2.5k.xes',
        'IRO2.5k.xes',
        'lp2.5k.xes',
        'OIR2.5k.xes',
        'ORI2.5k.xes',
        'pl2.5k.xes',
        'pm2.5k.xes',
        're2.5k.xes',
        'RIO2.5k.xes',
        'ROI2.5k.xes',
        'rp2.5k.xes',
        'sw2.5k.xes',
    ]
    lognames5000 = [
        'cb5k.xes',
        'cd5k.xes',
        'cf5k.xes',
        'cm5k.xes',
        'cp5k.xes',
        # 'fr5k.xes',
        'IOR5k.xes',
        'IRO5k.xes',
        'lp5k.xes',
        'OIR5k.xes',
        'ORI5k.xes',
        'pl5k.xes',
        'pm5k.xes',
        're5k.xes',
        'RIO5k.xes',
        'ROI5k.xes',
        'rp5k.xes',
        'sw5k.xes',
    ]
    lognames7500 = [
        'cb7.5k.xes',
        'cd7.5k.xes',
        'cf7.5k.xes',
        'cm7.5k.xes',
        'cp7.5k.xes',
        # 'fr7.5k.xes',
        'IOR7.5k.xes',
        'IRO7.5k.xes',
        'lp7.5k.xes',
        'OIR7.5k.xes',
        'ORI7.5k.xes',
        'pl7.5k.xes',
        'pm7.5k.xes',
        're7.5k.xes',
        'RIO7.5k.xes',
        'ROI7.5k.xes',
        'rp7.5k.xes',
        'sw7.5k.xes',
    ]
    lognames10000 = [
        'cb10k.xes',
        'cd10k.xes',
        'cf10k.xes',
        'cm10k.xes',
        'cp10k.xes',
        # 'fr10k.xes',
        'IOR10k.xes',
        'IRO10k.xes',
        'lp10k.xes',
        'OIR10k.xes',
        'ORI10k.xes',
        'pl10k.xes',
        'pm10k.xes',
        're10k.xes',
        'RIO10k.xes',
        'ROI10k.xes',
        'rp10k.xes',
        'sw10k.xes',
    ]

    lognames = lognames2500 + lognames5000 + lognames7500 + lognames10000
    # winsizes = [i for i in range(100, 1001, 100)]
    winsizes = [i for i in range(25, 301, 25)]
    deltas = [0.002, 0.05, 0.1, 0.3]

    # for testing one specific scenario
    # lognames = ['cb2.5k.xes']
    # winsizes = [100]
    # deltas = [0.002]

    ###############################################################
    # Information for calculating evaluation metrics
    ###############################################################
    actual_change_points = {
        '2.5k': define_change_points_dataset1(250),
        '5k': define_change_points_dataset1(500),
        '7.5k': define_change_points_dataset1(750),
        '10k': define_change_points_dataset1(1000)
    }

    # for files that do not follow the correct pattern
    exceptions_in_actual_change_points = {
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

    number_of_instances = {
        '2.5k': 2500,
        '5k': 5000,
        '7.5k': 7500,
        '10k': 10000
    }


class Dataset2Configuration:
    ###############################################################
    # Information about the data for performing the experiments
    ###############################################################
    input_folder = 'data/input/logs/Controlflow/dataset2'
    lognames3000 = [
        'cb3k.xes',
        'cd3k.xes',
        'cf3k.xes',
        'cm3k.xes',
        'cp3k.xes',
        'IOR3k.xes',
        'IRO3k.xes',
        'lp3k.xes',
        'OIR3k.xes',
        'ORI3k.xes',
        'pl3k.xes',
        'pm3k.xes',
        're3k.xes',
        'RIO3k.xes',
        'ROI3k.xes',
        'rp3k.xes',
        'sw3k.xes',
    ]
    lognames4500 = [
        'cb4.5k.xes',
        'cd4.5k.xes',
        'cf4.5k.xes',
        'cm4.5k.xes',
        'cp4.5k.xes',
        'IOR4.5k.xes',
        'IRO4.5k.xes',
        'lp4.5k.xes',
        'OIR4.5k.xes',
        'ORI4.5k.xes',
        'pl4.5k.xes',
        'pm4.5k.xes',
        're4.5k.xes',
        'RIO4.5k.xes',
        'ROI4.5k.xes',
        'rp4.5k.xes',
        'sw4.5k.xes',
    ]
    lognames8000 = [
        'cb8k.xes',
        'cd8k.xes',
        'cf8k.xes',
        'cm8k.xes',
        'cp8k.xes',
        'IOR8k.xes',
        'IRO8k.xes',
        'lp8k.xes',
        'OIR8k.xes',
        'ORI8k.xes',
        'pl8k.xes',
        'pm8k.xes',
        're8k.xes',
        'RIO8k.xes',
        'ROI8k.xes',
        'rp8k.xes',
        'sw8k.xes',
    ]

    lognames = lognames3000 + lognames4500 + lognames8000
    # winsizes = [i for i in range(100, 1001, 100)]
    winsizes = [i for i in range(25, 301, 25)]
    deltas = [0.002, 0.05, 0.1, 0.3]

    # for testing one specific scenario
    # lognames = ['cb2.5k.xes']
    # winsizes = [100]
    # deltas = [0.002]


def model_similarity_strategie_fixed_window(dataset_config, output_folder):
    metrics = [
        # SimilarityMetric.NODES,
        SimilarityMetric.EDGES
    ]

    # for testing
    # lognames = ['cb2.5k.xes']
    # windows = [200]
    # deltas = [0.002]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    drifts = dict.fromkeys(dataset_config.lognames)
    for log in dataset_config.lognames:
        drifts[log] = {}
        for w in dataset_config.winsizes:
            for d in dataset_config.deltas:
                drifts[log][f'{change_points_key}d={d} w={w}'] = apply_detector_on_model_similarity_fixed_window(
                    dataset_config.input_folder, log, metrics, d, w, output_folder, 100)

    df1 = pd.DataFrame.from_dict(drifts, orient='index')
    df1.to_excel(os.path.join(output_folder, 'experiments_model_similarity_fixed_window.xlsx'))


def quality_strategie_trace_by_trace(dataset_config, output_folder):
    # different metrics can be used for each dimension evaluated
    # by now we expected one metric for fitness quality dimension and other for precision quality dimension
    metrics = {
        QualityDimension.FITNESS.name: 'fitnessTBR',
        QualityDimension.PRECISION.name: 'precisionETC',
    }

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    drifts = dict.fromkeys(dataset_config.lognames)
    for log in dataset_config.lognames:
        drifts[log] = {}
        for sp in dataset_config.winsizes:
            for d in dataset_config.deltas:
                drifts[log][f'{change_points_key}d={d} sp={sp}'] = apply_detector_on_quality_metrics_trace_by_trace(
                    dataset_config.input_folder, log, metrics, d, sp, output_folder)

    df1 = pd.DataFrame.from_dict(drifts, orient='index')
    df1.to_excel(os.path.join(output_folder, 'experiments_quality_metrics_trace_by_trace.xlsx'))


def quality_strategie_fixed_window(dataset_config, output_folder):
    drifts = dict.fromkeys(dataset_config.lognames)
    for log in dataset_config.lognames:
        drifts[log] = {}
        for winsize in dataset_config.winsizes:
            for d in dataset_config.deltas:
                drifts[log][f'{change_points_key}d={d} w={winsize}'] = \
                    apply_detector_on_quality_metrics_fixed_window(dataset_config.input_folder, log, output_folder,
                                                                   winsize, d, 100)

    df1 = pd.DataFrame.from_dict(drifts, orient='index')
    df1.to_excel(os.path.join(output_folder, 'experiments_quality_metrics_fixed_window.xlsx'))


def quality_strategie_adaptive_window(dataset_config, output_folder):
    drifts = dict.fromkeys(dataset_config.lognames)
    for log in dataset_config.lognames:
        drifts[log] = {}
        for winsize in dataset_config.winsizes:
            # for d in config.deltas:
            #     drifts[log][f'{change_points_key}d={d} w={winsize}'] = \
            #         apply_detector_on_quality_metrics_adaptive_window(config.input_folder, log, output_folder, winsize, d, 100)
            d = 0.1
            drifts[log][f'{change_points_key}d={d} w={winsize}'] = \
                apply_detector_on_quality_metrics_adaptive_window(dataset_config.input_folder, log, output_folder,
                                                                  winsize, d, 100)
    df1 = pd.DataFrame.from_dict(drifts, orient='index')
    df1.to_excel(os.path.join(output_folder, 'experiments_quality_metrics_adaptive_window.xlsx'))


if __name__ == '__main__':
    #################################################################
    # EXPERIMENTS USING DATASET 1
    #################################################################
    # dataset_config = Dataset1Configuration()
    # output_folder = f'data/output/controlflow_adaptive/detection_on_quality_metrics_trace_by_trace/dataset1'
    # quality_strategie_trace_by_trace(dataset_config, output_folder)
    # output_folder = f'data/output/controlflow_adaptive/detection_on_quality_metrics_fixed_window/dataset1'
    # quality_strategie_fixed_window(dataset_config, output_folder)
    # output_folder = f'data/output/controlflow_adaptive/detection_on_model_similarity_fixed_window/dataset1'
    # model_similarity_strategie_fixed_window(dataset_config, output_folder)
    # output_folder = f'data/output/controlflow_adaptive/detection_on_quality_metrics_adaptive_window/dataset1'
    # quality_strategie_adaptive_window(dataset_config, output_folder)
    #################################################################
    # EXPERIMENTS USING DATASET 2
    #################################################################
    dataset_config = Dataset2Configuration()
    output_folder = f'data/output/controlflow_adaptive/detection_on_quality_metrics_trace_by_trace/dataset2'
    quality_strategie_trace_by_trace(dataset_config, output_folder)
    # output_folder = f'data/output/controlflow_adaptive/detection_on_quality_metrics_fixed_window/dataset2'
    # quality_strategie_fixed_window(dataset_config, output_folder)
    # output_folder = f'data/output/controlflow_adaptive/detection_on_model_similarity_fixed_window/dataset2'
    # model_similarity_strategie_fixed_window(dataset_config, output_folder)
    # output_folder = f'data/output/controlflow_adaptive/detection_on_quality_metrics_adaptive_window/dataset2'
    # quality_strategie_adaptive_window(dataset_config, output_folder)
