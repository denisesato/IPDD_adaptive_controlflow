import os
import pandas as pd


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


def calculate_metrics(metrics, detected_drifts, actual_drifts_informed, total_of_instances, et=0):
    real_drifts = actual_drifts_informed.copy()
    # sort the both lists (real and detected drifts)
    real_drifts.sort()
    detected_drifts.sort()

    # create lists to store the tp's and fp's
    tp_list = []
    fp_list = []
    total_distance = 0
    for detected_cp in detected_drifts:
        tp_found = False
        for real_cp in real_drifts:
            dist = detected_cp - real_cp
            if 0 <= dist <= et:
                total_distance += dist
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


def calculate_metrics_dataset1(filepath, filename, metrics, scenarios, actual_change_points, number_of_instances,
                               error_tolerance, save_input_for_calculation=False):
    input_filename = os.path.join(filepath, filename)
    print(f'*****************************************************************')
    print(f'Calculating metrics for file {input_filename}...')
    print(f'*****************************************************************')
    df = pd.read_excel(input_filename, index_col=0)
    complete_results = df.T.to_dict()
    metrics_results = {}
    for key in complete_results.keys():
        metrics_results[key] = {}
        scenario = [i for i in scenarios
                    if i in key][0]
        change_points = complete_results[key]
        for scenario_configuration in change_points.keys():
            # get detected drifts and convert to a list of integers
            detected_change_points = change_points[scenario_configuration][1:-1].split(",")
            detected_drifts = convert_list_to_int(detected_change_points)
            metrics = calculate_metrics(metrics, detected_drifts, actual_change_points[scenario],
                                        number_of_instances[scenario], error_tolerance[scenario])
            # add the calculated metrics to the dictionary
            if save_input_for_calculation:
                metrics_results[key][f'Detected drifts {scenario_configuration}'] = detected_drifts
                metrics_results[key][f'Real drifts {scenario_configuration}'] = actual_change_points[scenario]
            # print(f'-----------------------------------------------------------------')
            # print(f'Scenario: {key} - {scenario} - {delta}')
            # print(f'Real change points = {actual_change_points[scenario]}')
            # print(f'Error tolerance = {error_tolerance[scenario]}')
            # print(f'Detected change points = {detected_drifts}')
            for m in metrics:
                metrics_results[key][f'{m} {scenario_configuration}'] = metrics[m]
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

