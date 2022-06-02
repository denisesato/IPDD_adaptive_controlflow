import os

import pandas as pd
import matplotlib.pyplot as plt
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.algo.evaluation.generalization import algorithm as generalization_evaluator
from pm4py.objects.log.obj import EventLog
from pm4py.objects.log.util import interval_lifecycle
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py import discover_directly_follows_graph
from pm4py.visualization.dfg import visualizer as dfg_visualization
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.statistics.variants.log import get as variants_module
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
from pm4py.algo.evaluation.earth_mover_distance import algorithm as emd_evaluator
from pm4py.algo.conformance.alignments.edit_distance import algorithm as logs_alignments
from pm4py.evaluation.replay_fitness.variants.alignment_based import evaluate
from pm4py.algo.conformance.footprints.util import evaluation
from pm4py.algo.discovery.footprints import algorithm as fp_discovery
from enum import Enum

from skmultiflow.drift_detection import ADWIN

from compare_dfg import calculate_nodes_similarity, calculate_edges_similarity


class QualityDimension(str, Enum):
    FITNESS = 'fitness'
    PRECISION = 'precision'
    GENERALIZATION = 'generalization'


class SimilarityMetric(str, Enum):
    NODES = 'nodes similarity'
    EDGES = 'edges similarity'


# plot the values calculated for both quality dimensions and indicate the detected drifts
def save_plot(metrics, values, output_folder, output_name, drifts, similarity=False):
    plt.style.use('seaborn-whitegrid')
    plt.clf()
    if not similarity:  # used when defining quality metrics
        for metric in metrics.keys():
            plt.plot(values[metric], label=metrics[metric])
            no_values = len(values[metric])
    else:
        for metric in metrics:
            plt.plot(values[metric.name], label=metric.value)
            no_values = len(values[metric.name])
    gap = int(no_values * 0.1)
    if gap == 0:  # less than 10 values
        gap = 1
    xpos = range(0, no_values + 1, gap)

    # draw a line for each reported drift
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


def calculate_quality_metric_footprints(metric_name, log, tree):
    if metric_name == 'precisionFP':
        fp_log = fp_discovery.apply(log, variant=fp_discovery.Variants.TRACE_BY_TRACE)
        fp_tree = fp_discovery.apply(tree, variant=fp_discovery.Variants.PROCESS_TREE)
        precision = evaluation.fp_precision(fp_log, fp_tree)
        return precision


def calculate_quality_metric(metric_name, log, net, im, fm):
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
    elif metric_name == 'EMD':
        language = variants_module.get_language(log)
        playout_log = simulator.apply(net, im, fm,
                                      parameters={simulator.Variants.STOCHASTIC_PLAYOUT.value.Parameters.LOG: log},
                                      variant=simulator.Variants.STOCHASTIC_PLAYOUT)
        model_language = variants_module.get_language(playout_log)
        emd = emd_evaluator.apply(model_language, language)
        return emd
    elif metric_name == 'log_alignments':
        simulated_log = simulator.apply(net, im, fm)
        parameters = {}
        alignments = logs_alignments.apply(log, simulated_log, parameters=parameters)
        fitness = evaluate(alignments)
        return fitness['average_trace_fitness']
    else:
        print(f'metric name not identified {metric_name} in calculate_metric')
        return 0


# calculate the similarity between the nodes and edges between the initial model and the current one
# the current model is discovered using all the traces read since the last drift
# the initial model is discovered using the stable period (after a drift this model is updated)
# use the list of nodes and edges (obtained from the DFG)
def calculate_similarity_metric(m, initial_nodes, initial_edges, current_nodes, current_edges):
    value = 0
    if m == SimilarityMetric.NODES:
        value, added, removed = calculate_nodes_similarity(initial_nodes, current_nodes)
    elif m == SimilarityMetric.EDGES:
        value, added, removed = calculate_edges_similarity(initial_edges, current_edges)
    else:
        print(f'Similarity metric {m} is not implemented!')
    return value, added, removed


def calculate_edges_similarity_metric(m, initial_edges, current_edges):
    value, added, removed = calculate_edges_similarity(initial_edges, current_edges)
    return value, added, removed


# Apply the ADWIN detector (scikit-multiflow) in two quality dimensions: fitness and precision
# The metrics for each dimension are defined by parameter metrics (dictionary)
# The metrics are calculated using the last trace read and the model generated using the first traces (stable_period)
# When a drift is detected a new model may be discovered using the next traces (stable_period)
# The process model is discovered using the inductive miner
def apply_detector_on_quality_metrics_trace_by_trace(folder, logname, metrics, delta_detection, stable_period,
                                                     output_folder, update_model=True):
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
    # other discovery algorithms can be applied
    # net, im, fm = heuristics_miner.apply(log_for_model)
    # net, im, fm = inductive_miner.apply(log_for_model, variant=inductive_miner.Variants.IMf)
    # net, im, fm = inductive_miner.apply(log_for_model, variant=inductive_miner.Variants.IMd)
    gviz_pn = pn_visualizer.apply(net, im, fm)
    pn_visualizer.save(gviz_pn,
                       os.path.join(output_folder, f'{logname}_PN_INITIAL_0_{stable_period - 1}.png'))

    adwin_detection = {}
    drifts = {}
    values = {}
    for dimension in metrics.keys():
        # instantiate one detector for each evaluated dimension (fitness and precision)
        adwin_detection[dimension] = ADWIN(delta=delta_detection)
        drifts[dimension] = []
        values[dimension] = []

    model_no = 1
    total_of_traces = len(eventlog)
    initial_trace_id = 0
    final_trace_id = initial_trace_id + stable_period
    for i in range(0, total_of_traces):
        print(f'Reading trace [{i}]...')
        last_trace = EventLog(eventlog[i:(i + 1)])

        # check if one of the metrics report a drift
        drift_detected = False
        for dimension in metrics.keys():
            # calculate the metric for each dimension
            # for each dimension decide if the metric should be calculated using only the last trace read or all
            # the traces read since the last drift
            new_value = calculate_quality_metric(metrics[dimension], last_trace, net, im, fm) * 100
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
            # discover a new model using the next traces (stable_period)
            final_trace_id = i + stable_period
            if final_trace_id > total_of_traces:
                final_trace_id = total_of_traces

            if update_model:
                print(f'Discover a new model using traces from {i} to {final_trace_id - 1}')
                log_for_model = EventLog(eventlog[i:final_trace_id])
                net, im, fm = inductive_miner.apply(log_for_model)
                # other discovery algorithms can be applied
                # net, im, fm = heuristics_miner.apply(log_for_model)
                # net, im, fm = inductive_miner.apply(log_for_model, variant=inductive_miner.Variants.IMf)
                # net, im, fm = inductive_miner.apply(log_for_model, variant=inductive_miner.Variants.IMd)
                gviz_pn = pn_visualizer.apply(net, im, fm)
                pn_visualizer.save(gviz_pn,
                                   os.path.join(output_folder, f'{logname}_PN_{model_no}_{i}_{final_trace_id - 1}.png'))
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


# Apply the ADWIN detector (scikit-multiflow) in the similarity metrics informed
# (nodes and edges similarity implemented)
# The window_size define the number of traces to discover the initial process models
# We apply a sliding window (after reding a new trace the oldest is removed and the new one is included in the window)
# We discover a model using the updated window and compare it to the initial one
# The calculated similarity metrics are inputted in the ADWIN detector and if a drift is detected
# a new initial model is discovered
# The change point is the initial trace of the window when the similarity informed that a behavior is removed
# Otherwise the change point is the current trace
def apply_detector_on_model_similarity_fixed_window(folder, logname, metrics, delta_detection, window_size,
                                                    output_folder, factor):
    models_path = os.path.join(output_folder, 'models')
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    # for debug
    # debug_path = os.path.join(output_folder, 'models_for_debug')
    # if not os.path.exists(debug_path):
    #     os.makedirs(debug_path)

    # import the event log sorted by timestamp
    variant = xes_importer.Variants.ITERPARSE
    parameters = {variant.value.Parameters.TIMESTAMP_SORT: True}
    original_eventlog = xes_importer.apply(os.path.join(folder, logname), variant=variant, parameters=parameters)
    # convert to interval log, if no interval log is provided as input this line has no effect
    eventlog = interval_lifecycle.to_interval(original_eventlog)
    # derive the initial model using the parameter stable_period
    print(f'Initial model discovered using traces from 0 to {window_size - 1}')
    base_window = EventLog(eventlog[0:window_size])
    # get the activities of the log
    base_activities = list(attributes_filter.get_attribute_values(base_window, "concept:name").keys())
    # mine the DFG (using Pm4Py)
    base_dfg, start_activities, end_activities = discover_directly_follows_graph(base_window)
    parameters = {dfg_visualization.Variants.FREQUENCY.value.Parameters.START_ACTIVITIES: start_activities,
                  dfg_visualization.Variants.FREQUENCY.value.Parameters.END_ACTIVITIES: end_activities}
    model_no = 1
    gviz = dfg_visualization.apply(base_dfg, log=base_window, parameters=parameters)
    dfg_visualization.save(gviz, os.path.join(models_path,
                                              f'{logname}_w{window_size}_d{delta_detection}_DFG{model_no}_[0-{window_size - 1}].png'))

    adwin_detection = {}
    drifts = {}
    values = {}

    for m in metrics:
        # instantiate one detector for each similarity metric
        adwin_detection[m.name] = ADWIN(delta=delta_detection)
        drifts[m.name] = []
        values[m.name] = []
        for i in range(0, window_size):
            values[m.name].append(factor)
            adwin_detection[m.name].add_element(factor)
    # fill the initial values for the initial model
    print(f'Added {window_size} values for similarity during the stable period')

    total_of_traces = len(eventlog)
    initial_trace_id = 0
    for i in range(window_size, total_of_traces):
        if i >= initial_trace_id + window_size:
            print(f'Reading trace [{i}] - Current model [{i - window_size + 1}-{i}]')
            current_window = EventLog(eventlog[i - window_size + 1:(i + 1)])
            # get the current activities
            current_activities = list(attributes_filter.get_attribute_values(current_window, "concept:name").keys())
            # get the current edges from the directly-follows graph
            current_dfg, current_start_activities, current_end_activities = discover_directly_follows_graph(
                current_window)

            # for debug save the current models
            # gviz = dfg_visualization.apply(current_dfg, log=current_window, parameters=parameters)
            # dfg_visualization.save(gviz, os.path.join(debug_path,
            #                                           f'{logname}_CurrentDFG_[{i-window_size+1}-{i}]_.png'))

            # check if one of the metrics report a drift
            drift_detected = False
            for m in metrics:
                # calculate all the defined metrics
                new_value, added, removed = calculate_similarity_metric(m, base_activities, base_dfg,
                                                                        current_activities,
                                                                        current_dfg)
                new_value = new_value * factor
                print(f'Metric [{m}] - {new_value} - added {added} - removed {removed}')
                values[m.name].append(new_value)
                # update the new value in the detector
                adwin_detection[m.name].add_element(new_value)
                if adwin_detection[m.name].detected_change():
                    if len(removed) > 0:
                        # if something is removed from the previous model, assume the beginning
                        # of the window as the change point
                        change_point = i - window_size
                    else:
                        # if new behavior have been detected assume the end of the current window as the change point
                        change_point = i
                    # drift detected, save it
                    drifts[m.name].append(change_point)
                    print(f'Metric [{m.value}] - Drift detected at trace {i}')
                    print(f'Change point set to {change_point}')
                    # print(f'Added {added} - Removed {removed}')
                    drift_detected = True

            # if at least one metric report a drift a new model is discovered
            if drift_detected:
                for m in metrics:
                    # reset the detectors to avoid a new drift during the stable period
                    adwin_detection[m.name].reset()

                # Replace the base models using the current one
                base_window = current_window
                base_activities = current_activities
                base_dfg = current_dfg
                # save the new model
                model_no += 1
                gviz = dfg_visualization.apply(base_dfg, log=base_window, parameters=parameters)
                dfg_visualization.save(gviz, os.path.join(models_path,
                                                          f'{logname}_w{window_size}_d{delta_detection}_DFG{model_no}_[{i - window_size + 1}-{i}].png'))

    print(f'Size of values: {len(values[metrics[0].name])}')
    all_drifts = []
    for m in metrics:
        all_drifts += drifts[m.name]
        df = pd.DataFrame(values[m.name])
        df.to_excel(os.path.join(output_folder, f'{logname}_{m.value}_d{delta_detection}_w{window_size}.xlsx'))
    all_drifts = list(set(all_drifts))
    all_drifts.sort()
    save_plot(metrics, values, output_folder, f'{logname}_d{delta_detection}_w{window_size}', all_drifts,
              similarity=True)
    return all_drifts


# Calculate the metrics fitness and precision for detecting drifts
# Generate an initial model using the initial window (first traces - winsize)
# Read the next trace and update the window (remove the oldest and add the new one)
# Calculate the fitness using the current trace and the precision using the window, and the initial mode
# Input the new metrics in the ADWIN detector
# If the precision reports a drift inform the initial trace of the window as the change point
# If the fitness reports a drift inform the current trace of the window as the change point
# Restart the process updating the initial model
def apply_detector_on_quality_metrics_fixed_window(folder, logname, output_folder, winsize, delta=None, factor=1):
    # import the event log sorted by timestamp
    variant = xes_importer.Variants.ITERPARSE
    parameters = {variant.value.Parameters.TIMESTAMP_SORT: True}
    original_eventlog = xes_importer.apply(os.path.join(folder, logname), variant=variant, parameters=parameters)
    # convert to interval log, if no interval log is provided as input this line has no effect
    eventlog = interval_lifecycle.to_interval(original_eventlog)
    log_size = len(eventlog)
    # derive the model for evaluating the quality metrics
    initial_trace = 0
    log_for_model = EventLog(eventlog[initial_trace:winsize])
    net, im, fm = inductive_miner.apply(log_for_model)
    tree = inductive_miner.apply_tree(log_for_model)
    print(f'Initial model discovered using traces [{initial_trace}-{winsize - 1}]')
    model_number = 1
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    gviz_pn = pn_visualizer.apply(net, im, fm)
    models_folder = f'models_win{winsize}_factor{factor}'
    if delta:
        models_folder = f'{models_folder}_delta{delta}'
    models_output_path = os.path.join(output_folder, models_folder)
    if not os.path.exists(models_output_path):
        os.makedirs(models_output_path)
    pn_visualizer.save(gviz_pn, os.path.join(models_output_path,
                                             f'{logname}_PN_{model_number}_[{initial_trace}-{winsize - 1}].png'))

    metrics = {
        QualityDimension.FITNESS.name: 'fitnessTBR',
        QualityDimension.PRECISION.name: 'precisionFP'
    }

    values = dict.fromkeys(metrics)
    adwin = dict.fromkeys(metrics)
    drifts = dict.fromkeys(metrics)
    for m in metrics.keys():
        values[m] = []
        if delta:
            adwin[m] = ADWIN(delta=delta)
        else:
            adwin[m] = ADWIN()
        drifts[m] = []

    for i in range(0, log_size):
        # print(f'Reading trace {i}')
        current_trace = EventLog(eventlog[i:i + 1])
        if i == initial_trace:
            print(f'Setup phase - traces [{initial_trace}-{initial_trace + winsize - 1}]')
            # initial of the stable period
            # during the stable period we apply the same value for the metrics
            # fitness - calculated using the initial trace of the stable period
            # precision - calculated using all the traces inside the stable period
            traces_stable_period = EventLog(eventlog[initial_trace:initial_trace + winsize])
            # precision = calculate_metric(metrics[QualityDimension.PRECISION.name], traces_stable_period, net, im, fm)
            precision = calculate_quality_metric_footprints(metrics[QualityDimension.PRECISION.name],
                                                            traces_stable_period,
                                                            tree) * factor
            fitness = calculate_quality_metric(metrics[QualityDimension.FITNESS.name], current_trace, net, im,
                                               fm) * factor
        elif i >= initial_trace + winsize:
            print(f'Detection phase - reading trace {i}')
            window = EventLog(eventlog[i - winsize + 1:i + 1])
            # after the stable period calculate the metrics after reading a new trace
            # precision = calculate_metric(metrics[QualityDimension.PRECISION.name], window, net, im, fm)
            precision = calculate_quality_metric_footprints(metrics[QualityDimension.PRECISION.name], window,
                                                            tree) * factor
            fitness = calculate_quality_metric(metrics[QualityDimension.FITNESS.name], current_trace, net, im,
                                               fm) * factor

        values[QualityDimension.PRECISION.name].append(precision)
        adwin[QualityDimension.PRECISION.name].add_element(precision)

        values[QualityDimension.FITNESS.name].append(fitness)
        adwin[QualityDimension.FITNESS.name].add_element(fitness)

        drift_detected = False
        change_point = 0
        # check for drift in precision
        if adwin[QualityDimension.PRECISION.name].detected_change():
            # define the change point as the initial of the window
            change_point = i - winsize + 1
            drifts[QualityDimension.PRECISION.name].append(change_point)
            print(f'Metric [{QualityDimension.PRECISION.value}] detected a drift in trace: {change_point}')
            drift_detected = True
        # check for drift in fitness
        elif adwin[QualityDimension.FITNESS.name].detected_change():
            change_point = i
            drifts[QualityDimension.FITNESS.name].append(change_point)
            print(f'Metric [{QualityDimension.FITNESS.value}] detected a drift in trace: {change_point}')
            drift_detected = True

        if drift_detected:
            for m in metrics:
                # reset the detectors to avoid a new drift during the stable period
                adwin[m].reset()
            initial_trace = i + 1
            # Discover a new model using window
            model_number += 1
            log_for_model = EventLog(eventlog[change_point:change_point + winsize])
            net, im, fm = inductive_miner.apply(log_for_model)
            tree = inductive_miner.apply_tree(log_for_model)
            print(f'New model discovered using traces [{change_point}-{change_point + winsize - 1}]')
            gviz_pn = pn_visualizer.apply(net, im, fm)
            pn_visualizer.save(gviz_pn, os.path.join(models_output_path,
                                                     f'{logname}_PN_{model_number}_[{change_point}-{change_point + winsize - 1}].png'))

    print(f'Total of values for PRECISION: {len(values[QualityDimension.PRECISION.name])}')
    print(f'Total of values for FITNESS: {len(values[QualityDimension.FITNESS.name])}')

    all_drifts = []
    for m in metrics.keys():
        all_drifts += drifts[m]
        df = pd.DataFrame(values[m])
        df.to_excel(os.path.join(output_folder, f'{logname}_{m}_win{winsize}.xlsx'))
    all_drifts = list(set(all_drifts))
    all_drifts.sort()
    filename = f'{logname}_win{winsize}'
    if delta:
        filename = f'{filename}_delta{delta}'
    save_plot(metrics, values, output_folder, filename, all_drifts)
    return all_drifts


# Calculate the metrics fitness and precision for detecting drifts
# Generate an initial model using the initial window (first traces - winsize)
# Read the next trace and update the window (add the new one), the size of the window is adaptive
# Calculate the fitness using the current trace and the precision using the window,
# and the initial mode
# Input the new metrics in the ADWIN detector
# If the precision reports a drift inform the initial trace of the window as the change point
# If the fitness reports a drift inform the current trace of the window as the change point
# Restart the process updating the initial model
def apply_detector_on_quality_metrics_adaptive_window(folder, logname, output_folder, winsize, delta=None, factor=1):
    # import the event log sorted by timestamp
    variant = xes_importer.Variants.ITERPARSE
    parameters = {variant.value.Parameters.TIMESTAMP_SORT: True}
    original_eventlog = xes_importer.apply(os.path.join(folder, logname), variant=variant, parameters=parameters)
    # convert to interval log, if no interval log is provided as input this line has no effect
    eventlog = interval_lifecycle.to_interval(original_eventlog)
    log_size = len(eventlog)
    # derive the model for evaluating the quality metrics
    initial_trace = 0
    log_for_model = EventLog(eventlog[initial_trace:initial_trace+winsize])
    net, im, fm = inductive_miner.apply(log_for_model)
    tree = inductive_miner.apply_tree(log_for_model)

    #### Included for comparing models to avoid false alarms
    # mine the DFG (Pm4PY) for comparing similarity to avoid false alarms
    base_dfg, start_activities, end_activities = discover_directly_follows_graph(log_for_model)

    print(f'Initial model discovered using traces [{initial_trace}-{winsize-1}]')
    model_number = 1
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    gviz_pn = pn_visualizer.apply(net, im, fm)
    models_folder = f'models_win{winsize}_factor{factor}'
    if delta:
        models_folder = f'{models_folder}_delta{delta}'
    models_output_path = os.path.join(output_folder, models_folder)
    if not os.path.exists(models_output_path):
        os.makedirs(models_output_path)
    pn_visualizer.save(gviz_pn, os.path.join(models_output_path,
                                             f'{logname}_PN_{model_number}_[{initial_trace}-{initial_trace+winsize-1}].png'))
    gviz = dfg_visualization.apply(base_dfg, parameters=parameters)
    dfg_visualization.save(gviz, os.path.join(models_output_path,
                                              f'{logname}_DFG_{model_number}_[{initial_trace}-{initial_trace+winsize-1}].png'))

    metrics = {
        QualityDimension.FITNESS.name: 'fitnessTBR',
        QualityDimension.PRECISION.name: 'precisionETC'
    }

    values = dict.fromkeys(metrics)
    adwin = dict.fromkeys(metrics)
    drifts = dict.fromkeys(metrics)
    drifts_discarded_by_similarity = []
    for m in metrics.keys():
        values[m] = []
        if delta:
            adwin[m] = ADWIN(delta=delta)
        else:
            adwin[m] = ADWIN()
        drifts[m] = []

    for i in range(0, log_size):
        # print(f'Reading trace {i}')
        current_trace = EventLog(eventlog[i:i + 1])
        if i == initial_trace:
            print(f'Setup phase - traces [{initial_trace}-{initial_trace+winsize-1}]')
            # initial of the stable period
            # during the stable period we apply the same value for the metrics
            # fitness - calculated using the initial trace of the stable period
            # precision - calculated using all the traces inside the stable period
            traces_stable_period = EventLog(eventlog[initial_trace:initial_trace+winsize])
            precision = calculate_quality_metric(metrics[QualityDimension.PRECISION.name], traces_stable_period,
                                                            net, im, fm) * factor
            fitness = calculate_quality_metric(metrics[QualityDimension.FITNESS.name], current_trace, net, im,
                                               fm) * factor
        elif i >= initial_trace+winsize:
            print(f'Detection phase - reading trace {i} - window [{initial_trace}-{i}]')
            window = EventLog(eventlog[initial_trace:i+1])
            # after the stable period calculate the metrics after reading a new trace
            precision = calculate_quality_metric(metrics[QualityDimension.PRECISION.name], window,
                                                            net, im, fm) * factor
            fitness = calculate_quality_metric(metrics[QualityDimension.FITNESS.name], current_trace, net, im,
                                               fm) * factor

        values[QualityDimension.PRECISION.name].append(precision)
        adwin[QualityDimension.PRECISION.name].add_element(precision)

        values[QualityDimension.FITNESS.name].append(fitness)
        adwin[QualityDimension.FITNESS.name].add_element(fitness)

        drift_detected = False
        change_point = 0
        # check for drift in the calculated metrics
        for m in metrics.keys():
            if adwin[m].detected_change():
                # define the change point using the current trace
                change_point = i
                drifts[m].append(change_point)
                print(f'Metric [{metrics[m]}] detected a drift in trace: {change_point}')
                drift_detected = True

        if drift_detected:
            # check if the two models are similar, if they are do not considered the drift
            new_log_for_model = EventLog(eventlog[change_point:change_point+winsize+1])
            # mine the DFG (Pm4PY)for comparing similarity to avoid false alarms
            new_dfg, new_sa, new_ea = discover_directly_follows_graph(new_log_for_model)
            # calculate the edges similarity
            value, added, removed = calculate_edges_similarity_metric(m, base_dfg, new_dfg)
            if value < 1:  # process the drift
                for m in metrics:
                    # reset the detectors to avoid a new drift during the stable period
                    adwin[m].reset()
                initial_trace = i + 1
                # Discover a new model using window
                model_number += 1
                log_for_model = EventLog(eventlog[change_point:change_point+winsize])
                net, im, fm = inductive_miner.apply(log_for_model)
                tree = inductive_miner.apply_tree(log_for_model)
                # mine the DFG (Pm4PY) for comparing similarity to avoid false alarms
                base_dfg, start_activities, end_activities = discover_directly_follows_graph(log_for_model)
                gviz = dfg_visualization.apply(base_dfg, parameters=parameters)
                dfg_visualization.save(gviz, os.path.join(models_output_path,
                                                          f'{logname}_DFG_{model_number}_[{change_point}-{change_point+winsize-1}].png'))
                print(f'New model discovered using traces [{change_point}-{change_point+winsize-1}]')
                gviz_pn = pn_visualizer.apply(net, im, fm)
                pn_visualizer.save(gviz_pn, os.path.join(models_output_path,
                                                         f'{logname}_PN_{model_number}_[{change_point}-{change_point+winsize-1}].png'))
            else:
                print(f'Drift not considered because the two models are similar!')
                drifts_discarded_by_similarity.append(change_point)

    print(f'Total of values for PRECISION: {len(values[QualityDimension.PRECISION.name])}')
    print(f'Total of values for FITNESS: {len(values[QualityDimension.FITNESS.name])}')

    # save the information about drifts by metric and the discarded drifts by similarity
    filename = os.path.join(output_folder, f'{logname}_drifts_win{winsize}.txt')
    textfile = open(filename, 'w')

    all_drifts = []
    for m in metrics.keys():
        drifts[m] = [drift for drift in drifts[m] if drift not in drifts_discarded_by_similarity]
        all_drifts += drifts[m]
        df = pd.DataFrame(values[m])
        df.to_excel(os.path.join(output_folder, f'{logname}_{m}_win{winsize}.xlsx'))
        listToStr = ' '.join([str(elem) for elem in drifts[m]])
        textfile.write(m + " - " + listToStr + "\n")

    listToStr = ' '.join([str(elem) for elem in drifts_discarded_by_similarity])
    textfile.write('Discarded drifts - ' + listToStr + "\n")
    textfile.close()

    all_drifts = list(set(all_drifts))
    all_drifts.sort()
    filename = f'{logname}_win{winsize}'
    if delta:
        filename = f'{filename}_delta{delta}'
    save_plot(metrics, values, output_folder, filename, all_drifts)
    return all_drifts
