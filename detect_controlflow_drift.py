import os
import skmultiflow
import matplotlib.pyplot as plt
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.log.obj import EventLog
from pm4py.objects.log.util import interval_lifecycle
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from enum import Enum


class MetricDimension(str, Enum):
    FITNESS = 'fitness'
    PRECISION = 'precision'


# plot the values calculated for both quality dimensions and indicate the detected drifts
def save_plot(metrics, values, output_folder, output_name, drifts):
    plt.style.use('seaborn-whitegrid')
    plt.clf()
    plt.plot(values[metrics[MetricDimension.FITNESS.name]], label=metrics[MetricDimension.FITNESS.value])
    plt.plot(values[metrics[MetricDimension.PRECISION.name]], label=metrics[MetricDimension.PRECISION.value])
    no_values = len(values[metrics[MetricDimension.FITNESS.name]])
    gap = int(no_values * 0.1)
    if gap == 0:  # less than 10 values
        gap = 1
    xpos = range(0, no_values + 1, gap)

    # draw a line for each reported drift by the fitness dimension
    indexes = [int(x) for x in drifts[metrics[0]]]
    for d in indexes:
        plt.axvline(x=d, label=d, color='k', linestyle=':')

    # draw a line for each reported drift by the precision dimension
    indexes = [int(x) for x in drifts[metrics[1]]]
    for d in indexes:
        plt.axvline(x=d, label=d, color='green', linestyle=':')

    plt.legend()
    if len(drifts[metrics[MetricDimension.FITNESS.name]]) > 0 or len(drifts[metrics[MetricDimension.PRECISION.name]]) > 0:
        plt.xlabel('Trace')
    else:
        plt.xlabel('Trace - no drifts detected')
    plt.xticks(xpos, xpos, rotation=90)
    plt.ylabel(f'Metric value')
    plt.title(f'{output_name}')
    output_name = os.path.join(output_folder, f'{output_name}.png')
    plt.savefig(output_name)


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
    else:
        print(f'metric name not identified {metric_name} in calculate_metric')
        return 0


# apply the ADWIN detector (scikit-multiflow) in two metrics: fitness and precision
# when a drift is detected a new model is discovered using the next traces
# the stable_period define the number of traces to discover the process models
# the inductive miner is applied
def apply_adwin_updating_model(folder, logname, delta_detection, stable_period, output_folder):
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
        MetricDimension.PRECISION.name: 'precisionETC'
    }
    adwin_detection = {}
    drifts = {}
    values = {}
    for m in metrics.keys():
        # instantiate one detector for each evaluated dimension (fitness and precision)
        adwin_detection[m] = skmultiflow.drift_detection.ADWIN(delta=delta_detection)
        drifts[m] = []
        values[m] = []

    model_no = 1
    total_of_traces = len(eventlog)
    for i in range(0, total_of_traces):
        print(f'Reading trace [{i}]...')
        last_trace = EventLog(eventlog[i:(i + 1)])

        # calculate the fitness metric
        fitness_metric_name = metrics[MetricDimension.FITNESS.name]
        fitness = calculate_metric(fitness_metric_name, last_trace, net, im, fm) * 100
        values[fitness_metric_name].append(fitness)

        # calculate the precision metric
        precision_metric_name = metrics[MetricDimension.PRECISION.name]
        precision = calculate_metric(precision_metric_name, last_trace, net, im, fm) * 100
        values[precision_metric_name].append(precision)

        # update the new values in the two drift detectors
        adwin_detection[fitness_metric_name].add_element(fitness)
        adwin_detection[precision_metric_name].add_element(precision)

        # check if one of the metrics report a drift
        drift_detected = False
        if adwin_detection[fitness_metric_name].detected_change():
            # drift detected, save it
            drifts[fitness_metric_name].append(i)
            print(f'Metric [{fitness_metric_name}] - Drift detected at trace {i + 1}')
            drift_detected = True

        if adwin_detection[precision_metric_name].detected_change():
            # drift detected, save it
            drifts[precision_metric_name].append(i)
            print(f'Metric [{precision_metric_name}] - Drift detected at trace {i + 1}')
            drift_detected = True

        # if at least one metric report a drift a new model is discovered
        if drift_detected:
            # reset both detectors, to avoid a new drift during the stable period
            adwin_detection[fitness_metric_name].reset()
            adwin_detection[precision_metric_name].reset()
            # using next traces to discover a new model using the next traces (stable_period)
            final_trace_id = i + stable_period
            if final_trace_id > total_of_traces:
                final_trace_id = total_of_traces - 1

            print(f'Discover a new model using traces from {i} to {final_trace_id}')
            log_for_model = EventLog(eventlog[i:(final_trace_id + 1)])
            net, im, fm = inductive_miner.apply(log_for_model)
            gviz_pn = pn_visualizer.apply(net, im, fm)
            pn_visualizer.save(gviz_pn,
                               os.path.join(output_folder, f'{logname}_PN_{model_no}_{i}_{final_trace_id}.png'))
            model_no += 1

    save_plot(metrics, values, output_folder, f'{logname}_d{delta_detection}_sp{stable_period}', drifts)
    all_drifts = list(set(drifts[metrics[MetricDimension.FITNESS.name]] + drifts[metrics[MetricDimension.PRECISION.name]]))
    all_drifts.sort()
    combined_drifts = {f'd={delta_detection} sp={stable_period}': all_drifts}
    return combined_drifts
