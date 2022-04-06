import os
import pandas as pd
from detect_controlflow_drift import apply_adwin_on_quality_metrics, QualityDimension, apply_adwin_on_model_similarity, \
    SimilarityMetric


def dataset1_similarity_strategie():
    input_folder = 'data/input/logs/controlflow/dataset1'

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
    windows = [i for i in range(100, 1000, 50)]
    deltas = [0.002, 0.02, 0.1, 0.2, 0.3, 0.5]

    # for testing
    # lognames = ['cb2.5k.xes']
    windows = [100]
    deltas = [0.002]
    metrics = [
        # SimilarityMetric.NODES,
        SimilarityMetric.EDGES
    ]

    output_folder = f'data/output/controlflow_adaptive/detection_on_model_similarity_updating_model'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    drifts = dict.fromkeys(lognames)
    for log in lognames:
        drifts[log] = {}
        for w in windows:
            for d in deltas:
                drifts[log][f'd={d} w={w}'] = apply_adwin_on_model_similarity(input_folder, log, metrics, d, w,
                                                                              output_folder)

    df1 = pd.DataFrame.from_dict(drifts, orient='index')
    df1.to_excel(os.path.join(output_folder, 'experiments_model_similarity_dataset1.xlsx'))


def dataset1_quality_strategie():
    input_folder = 'data/input/logs/controlflow/dataset1'

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
    stable_periods = [i for i in range(100, 1000, 50)]
    deltas = [0.002, 0.02, 0.1, 0.2, 0.3, 0.5]

    # different metrics can be used for each dimension evaluated
    # by now we expected on metric for fitness quality dimension and other for precision quality dimension
    metrics = {
        QualityDimension.FITNESS.name: 'fitnessTBR',
        QualityDimension.PRECISION.name: 'precisionETC',
        # QualityDimension.GENERALIZATION.name: 'generalization'
    }

    # for testing
    lognames = ['cd5k.xes']
    stable_periods = [100]
    deltas = [0.02]

    output_folder = f'data/output/controlflow_adaptive/detection_on_quality_metrics_updating_model'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    drifts = dict.fromkeys(lognames)
    for log in lognames:
        drifts[log] = {}
        for sp in stable_periods:
            for d in deltas:
                drifts[log][f'd={d} sp={sp}'] = apply_adwin_on_quality_metrics(input_folder, log, metrics, d, sp,
                                                                               output_folder)

    df1 = pd.DataFrame.from_dict(drifts, orient='index')
    df1.to_excel(os.path.join(output_folder, 'experiments_dataset1.xlsx'))


if __name__ == '__main__':
    dataset1_quality_strategie()
    # dataset1_similarity_strategie()
