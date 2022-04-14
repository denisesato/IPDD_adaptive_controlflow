import os
import pandas as pd

change_points_key = 'drifts - '
detected_at_key = 'detected at - '


def calculate_f_score(tp, fp, fn):
    if tp + fp > 0:
        precision = tp / (tp + fp)
        if tp + fn > 0:
            recall = tp / (tp + fn)
        if precision > 0 or recall > 0:
            f_score = 2 * ((precision * recall) / (precision + recall))
            return f_score
    return 0


def calculate_fpr(tn, fp):
    return fp / (fp + tn)


def calculate_mean_delay(total_distance, tp):
    if total_distance == 0:
        return 0
    return total_distance / tp


# Calculate the metrics F-score, mean delay, and FPR (false positive rate)
# The f-score consider a TP a drift reported after an actual drift + et (error tolerance)
# The mean delay is the average of the delta between the trace where the drift was detected and the actual drift
# The delays is the difference between the actual drift and the moment where it is detected
# If the moment of detection occurs after the change point it should be informed in the parameter detected_at_list
def calculate_metrics(metrics, detected_drifts, actual_drifts_informed, total_of_instances, et, detected_at_list=None):
    real_drifts = actual_drifts_informed.copy()
    # sort the both lists (real and detected drifts)
    real_drifts.sort()
    detected_drifts.sort()
    if detected_at_list:
        detected_at_list.sort()

    # create lists to store the tp's and fp's
    tp_list = []
    fp_list = []
    total_distance = 0
    for i, detected_cp in enumerate(detected_drifts):
        tp_found = False
        for real_cp in real_drifts:
            if detected_at_list:
                dist_detection = detected_at_list[i] - real_cp
            else:
                dist_detection = detected_cp - real_cp
            dist = detected_cp - real_cp
            if 0 <= dist <= et:
                total_distance += dist_detection
                tp_list.append(detected_cp)
                tp_found = True
                real_drifts.remove(real_cp)
                break
            elif dist < 0:
                break
        if not tp_found:
            fp_list.append(detected_cp)

    tp = len(tp_list)
    fp = len(fp_list)
    fn = len(real_drifts)  # list contains only the real drifts not correctly detected
    tn = total_of_instances - tp - fp - fn
    metrics_result = {}
    for m in metrics:
        if m == 'f_score':
            metrics_result[m] = calculate_f_score(tp, fp, fn)
        if m == 'FPR':
            metrics_result[m] = calculate_fpr(tn, fp)
        if m == 'mean_delay':
            metrics_result[m] = calculate_mean_delay(total_distance, tp)
    return metrics_result


def calculate_metrics_dataset1(filepath, filename, metrics, logsizes, actual_change_points, number_of_instances,
                               error_tolerance, save_input_for_calculation=False):
    input_filename = os.path.join(filepath, filename)
    print(f'*****************************************************************')
    print(f'Calculating metrics for file {input_filename}...')
    print(f'*****************************************************************')
    df = pd.read_excel(input_filename, index_col=0)
    complete_results = df.T.to_dict()
    metrics_results = {}
    for logname in complete_results.keys():
        metrics_results[logname] = {}
        logsize = [i for i in logsizes
                   if i in logname][0]

        change_points = {}
        detected_at = {}
        for key in complete_results[logname].keys():
            # get list of trace ids from excel and convert to a list of integers
            trace_ids_list = complete_results[logname][key][1:-1].split(",")
            trace_ids_list = convert_list_to_int(trace_ids_list)

            # insert into change points or detected points
            if change_points_key in key:
                configuration = key[len(change_points_key):]
                change_points[configuration] = trace_ids_list
            elif detected_at_key in key:
                configuration = key[len(detected_at_key):]
                detected_at[configuration] = trace_ids_list

        for configuration in change_points.keys():
            # get the detected at information if available and convert to a list of integers
            if len(detected_at) > 0:
                metrics = calculate_metrics(metrics, change_points[configuration], actual_change_points[logsize],
                                            number_of_instances[logsize], error_tolerance[logsize],
                                            detected_at[configuration])
            else:
                metrics = calculate_metrics(metrics, change_points[configuration], actual_change_points[logsize],
                                            number_of_instances[logsize], error_tolerance[logsize])
            # add the calculated metrics to the dictionary
            if save_input_for_calculation:
                metrics_results[logname][f'Detected drifts {configuration}'] = change_points[configuration]
                if len(detected_at) > 0:
                    metrics_results[logname][f'Detected at {configuration}'] = detected_at[configuration]
                metrics_results[logname][f'Real drifts {configuration}'] = actual_change_points[logsize]
            # print(f'-----------------------------------------------------------------')
            # print(f'Scenario: {key} - {scenario} - {delta}')
            # print(f'Real change points = {actual_change_points[scenario]}')
            # print(f'Error tolerance = {error_tolerance[scenario]}')
            # print(f'Detected change points = {detected_drifts}')
            for m in metrics:
                metrics_results[logname][f'{m} {configuration}'] = metrics[m]
                # print(f'{m} {scenario_configuration} = {metrics[m]}')
            # print(f'-----------------------------------------------------------------')
    df = pd.DataFrame(metrics_results).T
    out_filename = filename[:-(len('.xlsx'))]
    out_filename = f'metrics_{out_filename}.xlsx'
    out_complete_filename = os.path.join(filepath, out_filename)
    print(f'*****************************************************************')
    print(f'Metrics for file {input_filename} calculated')
    print(f'Saving results at file {out_complete_filename}...')
    df.to_excel(out_complete_filename)
    print(f'*****************************************************************')


def convert_list_to_int(string_list):
    number_of_itens = len(string_list)
    integer_list = []
    if number_of_itens > 0 and string_list[0] != '':  # to avoid error in case of list with ''
        integer_map = map(int, string_list.copy())
        integer_list = list(integer_map)
    return integer_list
