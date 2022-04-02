import os

import pandas as pd
import skmultiflow
import matplotlib.pyplot as plt
from pm4py.algo.discovery.inductive.algorithm import Variants
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.objects.log.obj import EventLog
from pm4py.objects.log.util import interval_lifecycle
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from enum import Enum


class MetricDimension(str, Enum):
    FITNESS = 'fitness'
    PRECISION = 'precision'
    GENERALIZATION = 'generalization'


# plot the values calculated for both quality dimensions and indicate the detected drifts
def save_plot(metrics, values, output_folder, output_name, drifts):
    plt.style.use('seaborn-whitegrid')
    plt.clf()
    for metric in metrics.keys():
        plt.plot(values[metric], label=metrics[metric])
        no_values = len(values[metric])
    gap = int(no_values * 0.1)
    if gap == 0:  # less than 10 values
        gap = 1
    xpos = range(0, no_values + 1, gap)

    # draw a line for each reported drift by the fitness dimension
    indexes = [int(x) for x in drifts]
    for d in indexes:
        plt.axvline(x=d, label=d, color='k', linestyle=':')

    if len(drifts) > 0:
        plt.xlabel('Trace')
    else:
        plt.xlabel('Trace - no drifts detected')

    plt.xticks(xpos, xpos, rotation=90)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.ylabel(f'Metric value')
    plt.title(f'{output_name}')
    output_name = os.path.join(output_folder, f'{output_name}.png')
    plt.savefig(output_name, bbox_inches='tight')


def calculate_metric(metric_name, log, net, im, fm):
    if metric_name == 'precisionETC':
        return precision_evaluator.apply(log, net, im, fm,
                                         variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
    elif metric_name == 'precisionAL':
        precision = precision_evaluator.apply(log, net, im, fm,
                                              variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
        return precision
    elif metric_name == 'fitnessTBR':
        return replay_fitness_evaluator.apply(log, net, im, fm,
                                              variant=replay_fitness_evaluator.Variants.TOKEN_BASED)[
            'average_trace_fitness']
    elif metric_name == 'fitnessAL':
        fitness = replay_fitness_evaluator.apply(log, net, im, fm,
                                                 variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)
        return fitness['average_trace_fitness']
    elif metric_name == 'generalization':
        generalization = generalization_evaluator.apply(log, net, im, fm)
        return generalization
    else:
        print(f'metric name not identified {metric_name} in calculate_metric')
        return 0


# apply the ADWIN detector (scikit-multiflow) in two metrics: fitness and precision
# when a drift is detected a new model is discovered using the next traces
# the stable_period define the number of traces to discover the process models
# the inductive miner is applied
def apply_adwin_updating_model(folder, logname, delta_detection, stable_period, output_folder):
    output_folder = f'{output_folder}_d{delta_detection}_sp{stable_period}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # import the event log sorted by timestamp
    variant = xes_importer.Variants.ITERPARSE
    parameters = {variant.value.Parameters.TIMESTAMP_SORT: True}
    original_eventlog = xes_importer.apply(os.path.join(folder, logname), variant=variant, parameters=parameters)
    # convert to interval log, if no interval log is provided as input this line has no effect
    eventlog = interval_lifecycle.to_interval(original_eventlog)
    # derive the initial model using the parameter stable_period
    print(f'Initial model discovered using traces from 0 to {stable_period - 1}')
    log_for_model = EventLog(eventlog[0:stable_period])
    net, im, fm = inductive_miner.apply(log_for_model)
    gviz_pn = pn_visualizer.apply(net, im, fm)
    pn_visualizer.save(gviz_pn,
                       os.path.join(output_folder, f'{logname}_PN_INITIAL_0_{stable_period - 1}.png'))

    # different metrics can be used for each dimension evaluated
    # by now we expected on metric for fitness quality dimension and other for precision quality dimension
    metrics = {
        MetricDimension.FITNESS.name: 'fitnessTBR',
        MetricDimension.PRECISION.name: 'precisionETC',
    }
    adwin_detection = {}
    drifts = {}
    values = {}
    for dimension in metrics.keys():
        # instantiate one detector for each evaluated dimension (fitness and precision)
        adwin_detection[dimension] = skmultiflow.drift_detection.ADWIN(delta=delta_detection)
        drifts[dimension] = []
        values[dimension] = []

    model_no = 1
    total_of_traces = len(eventlog)
    initial_trace_id = 0
    final_trace_id = initial_trace_id + stable_period
    for i in range(0, total_of_traces):
        print(f'Reading trace [{i}]...')
        last_trace = EventLog(eventlog[i:(i+1)])
        if i >= stable_period:
            all_traces = EventLog(eventlog[initial_trace_id:(i+1)])
        else:
            all_traces = EventLog(eventlog[initial_trace_id:final_trace_id])
        # check if one of the metrics report a drift
        drift_detected = False
        for dimension in metrics.keys():
            # calculate the metric for each dimension
            new_value = calculate_metric(metrics[dimension], last_trace, net, im, fm) * 100
            values[dimension].append(new_value)
            # update the new value in the detector
            adwin_detection[dimension].add_element(new_value)
            if adwin_detection[dimension].detected_change():
                # drift detected, save it
                drifts[dimension].append(i)
                print(f'Metric [{dimension}] - Drift detected at trace {i}')
                drift_detected = True

        # if at least one metric report a drift a new model is discovered
        if drift_detected:
            for dimension in metrics.keys():
                # reset the detectors to avoid a new drift during the stable period
                adwin_detection[dimension].reset()
            initial_trace_id = i
            # discover a new model using the next traces (stable_period)
            final_trace_id = i + stable_period
            if final_trace_id > total_of_traces:
                final_trace_id = total_of_traces

            print(f'Discover a new model using traces from {i} to {final_trace_id-1}')
            log_for_model = EventLog(eventlog[i:final_trace_id])
            net, im, fm = inductive_miner.apply(log_for_model)
            gviz_pn = pn_visualizer.apply(net, im, fm)
            pn_visualizer.save(gviz_pn,
                               os.path.join(output_folder, f'{logname}_PN_{model_no}_{i}_{final_trace_id-1}.png'))
            model_no += 1

    all_drifts = []
    for dimension in metrics.keys():
        all_drifts += drifts[dimension]
        df = pd.DataFrame(values[dimension])
        df.to_excel(os.path.join(output_folder, f'{dimension}.xlsx'))
    all_drifts = list(set(all_drifts))
    all_drifts.sort()
    save_plot(metrics, values, output_folder, f'{logname}_d{delta_detection}_sp{stable_period}', all_drifts)
    return all_drifts


# apply the ADWIN detector (scikit-multiflow) in two metrics: fitness and precision
# fitness is calculated using the last trace read
# precision is calculated using all the traces read since the last detected drift
# the model is updated after each trace read
# when a drift is detected the current model is discarded
# the inductive miner is applied to discover the models
def apply_adwin_updating_model_after_each_trace(folder, logname, delta_detection, output_folder):
    output_folder = f'{output_folder}_d{delta_detection}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # import the event log sorted by timestamp
    variant = xes_importer.Variants.ITERPARSE
    parameters = {variant.value.Parameters.TIMESTAMP_SORT: True}
    original_eventlog = xes_importer.apply(os.path.join(folder, logname), variant=variant, parameters=parameters)
    # convert to interval log, if no interval log is provided as input this line has no effect
    eventlog = interval_lifecycle.to_interval(original_eventlog)

   # different metrics can be used for each dimension evaluated
    # by now we expected on metric for fitness quality dimension and other for precision quality dimension
    metrics = {
        # MetricDimension.FITNESS.name: 'fitnessTBR',
        MetricDimension.PRECISION.name: 'precisionETC',
        MetricDimension.GENERALIZATION.name: 'generalization'
    }
    adwin_detection = {}
    drifts = {}
    values = {}
    for dimension in metrics.keys():
        # instantiate one detector for each evaluated dimension (fitness and precision)
        adwin_detection[dimension] = skmultiflow.drift_detection.ADWIN(delta=delta_detection)
        drifts[dimension] = []
        values[dimension] = []

    total_of_traces = len(eventlog)
    initial_trace_id = 0
    model_no = 1
    for i in range(0, total_of_traces):
        print(f'Reading trace [{i}]...')
        # update the current model
        log_for_model = EventLog(eventlog[initial_trace_id:i+1])
        net, im, fm = inductive_miner.apply(log_for_model)
        gviz_pn = pn_visualizer.apply(net, im, fm)

        # for calculating the fitness
        last_trace = EventLog(eventlog[i:(i + 1)])

        # check if one of the metrics report a drift
        drift_detected = False
        for dimension in metrics.keys():
            # calculate the metric for each dimension
            metric_name = metrics[dimension]
            new_value = calculate_metric(metric_name, last_trace, net, im, fm) * 100
            values[dimension].append(new_value)

            # update the new value in the detector
            adwin_detection[dimension].add_element(new_value)

            if adwin_detection[dimension].detected_change():
                # drift detected, save it
                drifts[dimension].append(i)
                print(f'Metric [{dimension}] - Drift detected at trace {i}')
                drift_detected = True

        # if at least one metric report a drift a new model is discovered
        if drift_detected:
            # save the current model
            pn_visualizer.save(gviz_pn,
                               os.path.join(output_folder, f'{logname}_PN_{model_no}_{initial_trace_id}_{i}.png'))

            for dimension in metrics.keys():
                # reset the detectors to avoid a new drift during the stable period
                adwin_detection[dimension].reset()
            # reset the initial trace index considered for discovering the model
            initial_trace_id = i
            model_no += 1

    all_drifts = []
    for dimension in metrics.keys():
        all_drifts += drifts[dimension]
        df = pd.DataFrame(values[dimension])
        df.to_excel(os.path.join(output_folder, f'{dimension}.xlsx'))
    all_drifts = list(set(all_drifts))
    all_drifts.sort()
    save_plot(metrics, values, output_folder, f'{logname}_d{delta_detection}', all_drifts)
    return all_drifts
