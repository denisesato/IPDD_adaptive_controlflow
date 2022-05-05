import os
import pandas as pd

from evaluation_metrics import change_points_key
from detect_controlflow_drift import apply_detector_on_quality_metrics_trace_by_trace, QualityDimension, \
    SimilarityMetric, apply_detector_on_model_similarity_fixed_window, apply_detector_on_quality_metrics_fixed_window


class Dataset1Configuration:
    input_folder = 'data/input/logs/Controlflow/dataset1'

    lognames2500 = [
        'cb2.5k.xes',
        'cd2.5k.xes',
        'cf2.5k.xes',
        'cm2.5k.xes',
        'cp2.5k.xes',
        'fr2.5k.xes',
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
        'fr5k.xes',
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
        'fr7.5k.xes',
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
        'fr10k.xes',
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
    winsizes = [i for i in range(100, 1001, 100)]
    deltas = [0.002, 0.05, 0.1, 0.3]

    # for testing one specific scenario
    # lognames = ['cb2.5k.xes']
    # winsizes = [100]
    # deltas = [0.002]


def dataset1_similarity_strategie_fixed_window():
    config = Dataset1Configuration()
    metrics = [
        # SimilarityMetric.NODES,
        SimilarityMetric.EDGES
    ]

    # for testing
    # lognames = ['cb2.5k.xes']
    # windows = [200]
    # deltas = [0.002]

    output_folder = f'data/output/controlflow_adaptive/detection_on_model_similarity_fixed_window/dataset1'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    drifts = dict.fromkeys(config.lognames)
    for log in config.lognames:
        drifts[log] = {}
        for w in config.winsizes:
            for d in config.deltas:
                drifts[log][f'{change_points_key}d={d} w={w}'] = apply_detector_on_model_similarity_fixed_window(
                    config.input_folder, log, metrics, d, w, output_folder, 100)

    df1 = pd.DataFrame.from_dict(drifts, orient='index')
    df1.to_excel(os.path.join(output_folder, 'experiments_model_similarity_fixed_window_dataset1.xlsx'))


def dataset1_quality_strategie_trace_by_trace():
    config = Dataset1Configuration()

    # different metrics can be used for each dimension evaluated
    # by now we expected on metric for fitness quality dimension and other for precision quality dimension
    metrics = {
        QualityDimension.FITNESS.name: 'fitnessTBR',
        QualityDimension.PRECISION.name: 'precisionETC',
        # QualityDimension.GENERALIZATION.name: 'generalization'
    }

    output_folder = f'data/output/controlflow_adaptive/detection_on_quality_metrics_trace_by_trace/dataset1'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    drifts = dict.fromkeys(config.lognames)
    for log in config.lognames:
        drifts[log] = {}
        for sp in config.winsizes:
            for d in config.deltas:
                drifts[log][f'{change_points_key}d={d} sp={sp}'] = apply_detector_on_quality_metrics_trace_by_trace(
                    config.input_folder, log,
                    metrics, d, sp,
                    output_folder)

    df1 = pd.DataFrame.from_dict(drifts, orient='index')
    df1.to_excel(os.path.join(output_folder, 'experiments_quality_metrics_trace_by_trace_dataset1.xlsx'))


def dataset1_quality_strategie_fixed_window():
    config = Dataset1Configuration()
    output_folder = f'data/output/controlflow_adaptive/detection_on_quality_metrics_fixed_window/dataset1'
    drifts = dict.fromkeys(config.lognames)
    for log in config.lognames:
        drifts[log] = {}
        for winsize in config.winsizes:
            for d in config.deltas:

                drifts[log][f'{change_points_key}d={d} w={winsize}'] = \
                    apply_detector_on_quality_metrics_fixed_window(config.input_folder, log, output_folder, winsize, d, 100)

    df1 = pd.DataFrame.from_dict(drifts, orient='index')
    df1.to_excel(os.path.join(output_folder, 'experiments_quality_metrics_fixed_window_dataset1.xlsx'))


if __name__ == '__main__':
    # EXPERIMENTS USING DATASET 1
    dataset1_quality_strategie_trace_by_trace()
    dataset1_quality_strategie_fixed_window()
    dataset1_similarity_strategie_fixed_window()
