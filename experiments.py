import os
import pandas as pd

from calculate_evaluation_metrics import change_points_key
from detect_controlflow_drift import apply_detector_on_quality_metrics_trace_by_trace, QualityDimension, \
    apply_detector_on_model_similarity_fixed_window, \
    SimilarityMetric, apply_detector_on_quality_metrics_fixed_window, apply_detector_on_model_similarity_trace_by_trace, \
    apply_detector_on_model_similarity_fixed_window_NOVO, apply_detector_on_quality_metrics_fixed_window_TESTE


def dataset1_similarity_strategie_trace_by_trace():
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
    windows = [i for i in range(100, 1001, 100)]
    # deltas = [0.002, 0.02, 0.1, 0.2, 0.3, 0.5]
    deltas = [0.002]
    metrics = [
        # SimilarityMetric.NODES,
        SimilarityMetric.EDGES
    ]

    # for testing
    # lognames = ['ROI2.5k.xes']
    # windows = [100]
    # deltas = [0.002]


    output_folder = f'data/output/controlflow_adaptive/detection_on_model_similarity_fixed_trace_by_trace/dataset1'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    drifts = dict.fromkeys(lognames)
    for log in lognames:
        drifts[log] = {}
        for w in windows:
            for d in deltas:
                drifts[log][f'{change_points_key}d={d} w={w}'] = apply_detector_on_model_similarity_trace_by_trace(
                    input_folder, log, metrics, d, w, output_folder, 100)

    df1 = pd.DataFrame.from_dict(drifts, orient='index')
    df1.to_excel(os.path.join(output_folder, 'experiments_model_similarity_trace_by_trace_dataset1.xlsx'))


def dataset1_similarity_strategie_fixed_window():
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
    windows = [i for i in range(100, 1001, 100)]
    deltas = [0.002, 0.05, 0.1, 0.3]
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
    drifts = dict.fromkeys(lognames)
    for log in lognames:
        drifts[log] = {}
        for w in windows:
            for d in deltas:
                drifts[log][f'{change_points_key}d={d} w={w}'] = apply_detector_on_model_similarity_fixed_window(
                    input_folder, log, metrics, d, w, output_folder, 100)

    df1 = pd.DataFrame.from_dict(drifts, orient='index')
    df1.to_excel(os.path.join(output_folder, 'experiments_model_similarity_fixed_window_dataset1.xlsx'))


def dataset1_similarity_strategie_fixed_window_NOVO():
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
    windows = [i for i in range(100, 1001, 100)]
    deltas = [0.002, 0.05, 0.1, 0.3]
    metrics = [
        # SimilarityMetric.NODES,
        SimilarityMetric.EDGES
    ]

    # for testing
    # lognames = ['cb2.5k.xes']
    # windows = [200]
    # deltas = [0.002]

    output_folder = f'data/output/controlflow_adaptive/detection_on_model_similarity_fixed_window_NOVO/dataset1'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    drifts = dict.fromkeys(lognames)
    for log in lognames:
        drifts[log] = {}
        for w in windows:
            for d in deltas:
                drifts[log][f'{change_points_key}d={d} w={w}'] = apply_detector_on_model_similarity_fixed_window_NOVO(
                    input_folder, log, metrics, d, w, output_folder, 100)

    df1 = pd.DataFrame.from_dict(drifts, orient='index')
    df1.to_excel(os.path.join(output_folder, 'experiments_model_similarity_fixed_window_NOVO_dataset1.xlsx'))


def dataset1_quality_strategie_trace_by_trace():
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
    stable_periods = [i for i in range(100, 1001, 100)]
    deltas = [0.002, 0.05, 0.1, 0.3]

    # different metrics can be used for each dimension evaluated
    # by now we expected on metric for fitness quality dimension and other for precision quality dimension
    metrics = {
        QualityDimension.FITNESS.name: 'fitnessTBR',
        QualityDimension.PRECISION.name: 'precisionETC',
        # QualityDimension.GENERALIZATION.name: 'generalization'
    }

    # for testing
    # lognames = ['cd5k.xes']
    # stable_periods = [100]
    # deltas = [0.02]

    output_folder = f'data/output/controlflow_adaptive/detection_on_quality_metrics_trace_by_trace/dataset1'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    drifts = dict.fromkeys(lognames)
    for log in lognames:
        drifts[log] = {}
        for sp in stable_periods:
            for d in deltas:
                drifts[log][f'{change_points_key}d={d} sp={sp}'] = apply_detector_on_quality_metrics_trace_by_trace(input_folder, log,
                                                                                                 metrics, d, sp,
                                                                                                 output_folder)

    df1 = pd.DataFrame.from_dict(drifts, orient='index')
    df1.to_excel(os.path.join(output_folder, 'experiments_quality_trace_by_trace_dataset1.xlsx'))


def dataset1_quality_strategie_fixed_window():
    folder = 'data/input/logs/Controlflow/dataset1'
    output_folder = 'data/output/controlflow_adaptive/detection_on_quality_metrics_fixed_window/dataset1'

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

    # for testing
    # lognames = ['cd5k.xes']
    # winsizes = [200]
    # deltas = [0.002]

    drifts = dict.fromkeys(lognames)
    for log in lognames:
        drifts[log] = {}
        for winsize in winsizes:
            for d in deltas:
                # factor = 1
                # drifts[log][f'{change_points_key}d={d} w={winsize} f={factor}'] = \
                #     apply_detector_on_quality_metrics_fixed_window(folder, log, output_folder, winsize, d)
                # factor = 100
                # drifts[log][f'{change_points_key}d={d} w={winsize} f={factor}'] = \
                #     apply_detector_on_quality_metrics_fixed_window(folder, log, output_folder, winsize, d, factor)
                drifts[log][f'{change_points_key}d={d} w={winsize}'] = \
                    apply_detector_on_quality_metrics_fixed_window_TESTE(folder, log, output_folder, winsize, d, 100)

    df1 = pd.DataFrame.from_dict(drifts, orient='index')
    df1.to_excel(os.path.join(output_folder, 'experiments_quality_fixed_window_dataset1.xlsx'))


def dataset2_quality_strategie_fixed_window():
    folder = 'data/input/logs/Controlflow/dataset2'
    out_folder = 'data/output/windowing/dataset2'
    lognames = ['ConditionalMove.xes']
    # winsizes = [i for i in range(100, 1001, 100)]
    winsizes = [100]

    for logname in lognames:
        for winsize in winsizes:
            apply_detector_on_quality_metrics_fixed_window(folder, logname, out_folder, winsize, winsize)


def dataset3_similarity_strategie_fixed_window():
    input_folder = 'data/input/logs/controlflow/dataset3'

    logname_sudden_noise0_1000 = [
        'sudden_trace_noise0_1000_cb.xes',
        'sudden_trace_noise0_1000_cd.xes',
        'sudden_trace_noise0_1000_cf.xes',
        'sudden_trace_noise0_1000_cp.xes',
        'sudden_trace_noise0_1000_IOR.xes',
        'sudden_trace_noise0_1000_IRO.xes',
        'sudden_trace_noise0_1000_lp.xes',
        'sudden_trace_noise0_1000_OIR.xes',
        'sudden_trace_noise0_1000_pl.xes',
        'sudden_trace_noise0_1000_pm.xes',
        'sudden_trace_noise0_1000_re.xes',
        'sudden_trace_noise0_1000_RIO.xes',
        'sudden_trace_noise0_1000_ROI.xes',
        'sudden_trace_noise0_1000_rp.xes',
        'sudden_trace_noise0_1000_sw.xes',
    ]

    logname_sudden_noise5_100 = [
        'sudden_trace_noise5_100_cb.xes',
        'sudden_trace_noise5_100_cd.xes',
        'sudden_trace_noise5_100_cf.xes',
        'sudden_trace_noise5_100_cp.xes',
        'sudden_trace_noise5_100_IOR.xes',
        'sudden_trace_noise5_100_IRO.xes',
        'sudden_trace_noise5_100_lp.xes',
        'sudden_trace_noise5_100_OIR.xes',
        'sudden_trace_noise5_100_pl.xes',
        'sudden_trace_noise5_100_pm.xes',
        'sudden_trace_noise5_100_re.xes',
        'sudden_trace_noise5_100_RIO.xes',
        'sudden_trace_noise5_100_ROI.xes',
        'sudden_trace_noise5_100_rp.xes',
        'sudden_trace_noise5_100_sw.xes',
    ]

    logname_sudden_noise20_500 = [
        'sudden_trace_noise20_500_cb.xes',
        'sudden_trace_noise20_500_cd.xes',
        'sudden_trace_noise20_500_cf.xes',
        'sudden_trace_noise20_500_cp.xes',
        'sudden_trace_noise20_500_IOR.xes',
        'sudden_trace_noise20_500_IRO.xes',
        'sudden_trace_noise20_500_lp.xes',
        'sudden_trace_noise20_500_OIR.xes',
        'sudden_trace_noise20_500_pl.xes',
        'sudden_trace_noise20_500_pm.xes',
        'sudden_trace_noise20_500_re.xes',
        'sudden_trace_noise20_500_RIO.xes',
        'sudden_trace_noise20_500_ROI.xes',
        'sudden_trace_noise20_500_rp.xes',
        'sudden_trace_noise20_500_sw.xes',
    ]

    lognames = logname_sudden_noise0_1000 + logname_sudden_noise5_100
    windows = [i for i in range(100, 1000, 50)]
    deltas = [0.002, 0.02, 0.1, 0.2, 0.3, 0.5]

    # for testing
    # lognames = ['ROI2.5k.xes']
    lognames = logname_sudden_noise20_500
    windows = [20]
    deltas = [0.002]
    metrics = [
        SimilarityMetric.NODES,
        SimilarityMetric.EDGES
    ]

    output_folder = f'data/output/controlflow_adaptive/detection_on_model_similarity_fixed_window/dataset3'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    drifts = dict.fromkeys(lognames)
    for log in lognames:
        drifts[log] = {}
        for w in windows:
            for d in deltas:
                drifts[log][f'{change_points_key}d={d} w={w}'] = apply_detector_on_model_similarity_fixed_window(
                    input_folder, log, metrics, d, w, output_folder, 100)

    df1 = pd.DataFrame.from_dict(drifts, orient='index')
    df1.to_excel(os.path.join(output_folder, 'experiments_model_similarity_fixed_window_dataset3.xlsx'))


if __name__ == '__main__':
    # dataset3_similarity_strategie_fixed_window()

    # EXPERIMENTS USING DATASET 1
    # dataset1_quality_strategie_trace_by_trace()
    dataset1_quality_strategie_fixed_window()
    # dataset1_similarity_strategie_fixed_window()
    # dataset1_similarity_strategie_fixed_window_NOVO()
