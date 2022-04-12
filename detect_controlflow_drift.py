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


# apply the ADWIN detector (scikit-multiflow) in two quality metrics: fitness and precision
# the metrics are calculated using the last trace read and the model generated using the first traces (stable_period)
# it is possible to consider all the traces read since the last drift for calculating the metrics (commented)
# when a drift is detected a new model is discovered using the next traces (stable_period)
# the inductive miner is applied for discovering the model
def pply_detector_on_quality_metrics_trace_by_trace(folder, logname, metrics, delta_detection, stable_period, output_folder,
                                                    update_model=True):
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

        # get all the traces since started the log or since the last drift
        if i >= final_trace_id:
            all_traces_since_last_drift = EventLog(eventlog[initial_trace_id:(i + 1)])
        else:
            all_traces_since_last_drift = EventLog(eventlog[initial_trace_id:final_trace_id])

        # get the last n traces (n is defined by stable_period)
        if i-stable_period < initial_trace_id:
            # if we are still reading the traces from stable period
            # get the first traces from the stable period
            last_n_traces = EventLog(eventlog[initial_trace_id:stable_period])
            print(f'Last n traces considered [{initial_trace_id}-{stable_period-1}]')
        else:
            last_n_traces = EventLog(eventlog[i-stable_period:i])
            print(f'Last n traces considered [{i-stable_period}-{i-1}]')

        # check if one of the metrics report a drift
        drift_detected = False
        for dimension in metrics.keys():
            # calculate the metric for each dimension
            # for each dimension decide if the metric should be calculated using only the last trace read or all
            # the traces read since the last drift
            new_value = 0
            if dimension == QualityDimension.FITNESS.name:
                new_value = calculate_metric(metrics[dimension], last_trace, net, im, fm) * 100
            if dimension == QualityDimension.PRECISION.name:
                new_value = calculate_metric(metrics[dimension], last_trace, net, im, fm) * 100
                # new_value = calculate_metric(metrics[dimension], all_traces_since_last_drift, net, im, fm) * 100
                # new_value = calculate_metric(metrics[dimension], last_n_traces, net, im, fm) * 100
            if dimension == QualityDimension.GENERALIZATION.name:
                new_value = calculate_metric(metrics[dimension], last_trace, net, im, fm) * 100
                # new_value = calculate_metric(metrics[dimension], all_traces_since_last_drift, net, im, fm)
                # new_value = calculate_metric(metrics[dimension], last_n_traces, net, im, fm) * 100
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

            if update_model:
                print(f'Discover a new model using traces from {i} to {final_trace_id - 1}')
                log_for_model = EventLog(eventlog[i:final_trace_id])
                net, im, fm = inductive_miner.apply(log_for_model)
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
    return value


# apply the ADWIN detector (scikit-multiflow) in the two similarity metrics: nodes and edges similarity
# the stable_period define the number of traces to discover the initial process models
# we apply a sliding window on each new trace, discover the new model and compare to the initial one
# these metrics are inputted in the detector
# if a drift is detected a new initial model is discovered
def apply_detectir_on_model_similarity_fixed_window(folder, logname, metrics, delta_detection, window_size, output_folder):
    output_folder = f'{output_folder}_d{delta_detection}_w{window_size}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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
    dfg_base, start_activities, end_activities = discover_directly_follows_graph(base_window)
    parameters = {dfg_visualization.Variants.FREQUENCY.value.Parameters.START_ACTIVITIES: start_activities,
                  dfg_visualization.Variants.FREQUENCY.value.Parameters.END_ACTIVITIES: end_activities}
    model_no = 1
    gviz = dfg_visualization.apply(dfg_base, log=base_window, parameters=parameters)
    dfg_visualization.save(gviz, os.path.join(output_folder, f'{logname}_DFG{model_no}.png'))

    adwin_detection = {}
    drifts = {}
    values = {}

    for m in metrics:
        # instantiate one detector for each similarity metric
        adwin_detection[m.name] = ADWIN(delta=delta_detection)
        drifts[m.name] = []
        values[m.name] = []
        # fill the initial values for the initial model
        for i in range(0, window_size):
            values[m.name].append(100)

    total_of_traces = len(eventlog)
    for i in range(0, total_of_traces - window_size):
        print(f'Reading trace [{i}]...')
        current_window = EventLog(eventlog[i:(i + window_size)])
        # get the current activities
        current_activities = list(attributes_filter.get_attribute_values(current_window, "concept:name").keys())
        # get the current edges from the directly-follows graph
        current_dfg, current_start_activities, current_end_activities = discover_directly_follows_graph(current_window)

        # check if one of the metrics report a drift
        drift_detected = False
        for m in metrics:
            # calculate all the defined metrics
            new_value = calculate_similarity_metric(m, base_activities, dfg_base, current_activities,
                                                    current_dfg) * 100
            values[m.name].append(new_value)
            # update the new value in the detector
            adwin_detection[m.name].add_element(new_value)
            if adwin_detection[m.name].detected_change():
                change_point = i + window_size
                # drift detected, save it
                drifts[m.name].append(change_point)
                print(f'Metric [{m.value}] - Drift detected at trace {change_point}')
                drift_detected = True

        # if at least one metric report a drift a new model is discovered
        if drift_detected:
            for m in metrics:
                # reset the detectors to avoid a new drift during the stable period
                adwin_detection[m.name].reset()
            # discover a new model using the next traces (stable_period)
            base_window = EventLog(eventlog[i:(i + window_size)])
            # get the activities of the log
            base_activities = list(attributes_filter.get_attribute_values(base_window, "concept:name").keys())
            # mine the DFG (using Pm4Py)
            dfg_base, start_activities, end_activities = discover_directly_follows_graph(base_window)

            # save the new model
            model_no += 1
            gviz = dfg_visualization.apply(dfg_base, log=base_window, parameters=parameters)
            dfg_visualization.save(gviz, os.path.join(output_folder, f'{logname}_DFG{model_no}.png'))

    all_drifts = []
    for m in metrics:
        all_drifts += drifts[m.name]
        df = pd.DataFrame(values[m.name])
        df.to_excel(os.path.join(output_folder, f'{logname}_{m.value}.xlsx'))
    all_drifts = list(set(all_drifts))
    all_drifts.sort()
    save_plot(metrics, values, output_folder, f'{logname}_d{delta_detection}_sp{window_size}', all_drifts,
              similarity=True)
    return all_drifts


# nova estratégia, que utiliza janelamento para calcular fitness e precision
# o ADWIN é aplicado utilizando as 2 métricas, e, em caso
# de drift um novo modelo é gerado
def apply_detector_on_quality_metrics_fixed_window(folder, logname, output_folder, winsize, winstep, delta=None):
    # import the event log sorted by timestamp
    variant = xes_importer.Variants.ITERPARSE
    parameters = {variant.value.Parameters.TIMESTAMP_SORT: True}
    original_eventlog = xes_importer.apply(os.path.join(folder, logname), variant=variant, parameters=parameters)
    # convert to interval log, if no interval log is provided as input this line has no effect
    eventlog = interval_lifecycle.to_interval(original_eventlog)
    log_size = len(eventlog)
    # derive the model for evaluating the quality metrics
    log_for_model = EventLog(eventlog[0:winsize])
    net, im, fm = inductive_miner.apply(log_for_model)
    print(f'Initial model discovered using traces [0-{winsize-1}]')
    model_number = 1
    gviz_pn = pn_visualizer.apply(net, im, fm)
    models_output_path = os.path.join(output_folder, 'models')
    if not os.path.exists(models_output_path):
        os.makedirs(models_output_path)
    pn_visualizer.save(gviz_pn, os.path.join(models_output_path,
                                             f'{logname}_PN_{model_number}_0_{winsize-1}.png'))

    metrics = {
        QualityDimension.FITNESS.name: 'fitnessTBR',
        QualityDimension.PRECISION.name: 'precisionETC'
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

    for initial_trace in range(0, log_size - winstep + 1, winstep):
        print(f'Reading traces {initial_trace} to {initial_trace+winsize-1}')
        drift_detected = False
        change_point = 0
        window = EventLog(eventlog[initial_trace:initial_trace+winsize])
        precision = calculate_metric(metrics[QualityDimension.PRECISION.name], window, net, im, fm) * 100
        fitness = calculate_metric(metrics[QualityDimension.FITNESS.name], window, net, im, fm) * 100
        # fill the precision and fitness for the traces in the window with the calculated value
        for i in range(0, winstep):
            values[QualityDimension.PRECISION.name].append(precision)
            values[QualityDimension.FITNESS.name].append(fitness)
            # update the new values in the detector
            adwin[QualityDimension.PRECISION.name].add_element(precision)
            adwin[QualityDimension.FITNESS.name].add_element(fitness)
            # check for drift
            if adwin[QualityDimension.PRECISION.name].detected_change():
                if not drift_detected:
                    # first detection - save change point
                    change_point = initial_trace+winsize
                    drift_detected = True
                drifts[QualityDimension.PRECISION.name].append(change_point)
                print(f'Metric [{QualityDimension.PRECISION.value}] detected a drift in trace: {change_point}')
            # check for drift
            if adwin[QualityDimension.FITNESS.name].detected_change():
                if not drift_detected:
                    # first detection - save change point
                    change_point = initial_trace+winsize
                    drift_detected = True
                drifts[QualityDimension.FITNESS.name].append(change_point)
                print(f'Metric [{QualityDimension.FITNESS.value}] detected a drift in trace: {change_point}')

        if drift_detected:
            # Discover a new model using stable_period
            model_number += 1
            log_for_model = EventLog(eventlog[change_point:change_point+winsize])
            net, im, fm = inductive_miner.apply(log_for_model)
            print(f'New model discovered using traces [{change_point}-{change_point+winsize-1}]')
            gviz_pn = pn_visualizer.apply(net, im, fm)
            pn_visualizer.save(gviz_pn, os.path.join(models_output_path,
                                                     f'{logname}_PN_{model_number}_{change_point}_{change_point+winsize-1}.png'))

    print(f'Total of values for PRECISION: {len(values[QualityDimension.PRECISION.name])}')
    print(f'Total of values for FITNESS: {len(values[QualityDimension.FITNESS.name])}')

    all_drifts = []
    for m in metrics.keys():
        all_drifts += drifts[m]
        df = pd.DataFrame(values[m])
        df.to_excel(os.path.join(output_folder, f'{logname}_{m}.xlsx'))
    all_drifts = list(set(all_drifts))
    all_drifts.sort()
    filename = f'{logname}_win{winsize}_step{winstep}'
    save_plot(metrics, values, output_folder, filename, all_drifts)
