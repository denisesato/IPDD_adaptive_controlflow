import os
import re
import pandas as pd


def get_Apromore_files(log_file_path, key, ftype):
    # get the .txt files with the results reported by Apromore
    files = [i for i in os.listdir(log_file_path)
             if os.path.isfile(os.path.join(log_file_path, i))
             and i.endswith(ftype)
             and i.startswith(key)]

    return files


def convert_list_to_int(string_list):
    number_of_itens = len(string_list)
    integer_list = []
    if number_of_itens > 0 and string_list[0] != '':  # to avoid error in case of list with ''
        integer_map = map(int, string_list.copy())
        integer_list = list(integer_map)
    return integer_list


def read_drifts_prodrift(file):
    file = open(file, 'r')
    lines = file.readlines()

    reported_drifts = []
    reported_delays = []
    for line in lines:
        if line.startswith('('):
            change_point = line[line.index('trace: ') + len('trace: '):line.index(' (')]
            delay = line[line.index('reading ') + len('reading '):line.index(' traces.')]
            reported_drifts.append(change_point)
            reported_delays.append(delay)
    file.close()
    return convert_list_to_int(reported_drifts), convert_list_to_int(reported_delays)


def compile_results_from_prodrift(filepath, filenames):
    print(f'Looking for results...')
    results = {}

    for file in filenames:
        print(f'*****************************************************************')
        print(f'Reading file {file}...')
        print(f'*****************************************************************')
        complete_filename = os.path.join(filepath, file)
        pattern = 'log_([a-zA-Z]*)(.*?)_runs_(.*?)_(\d*).txt'
        if match := re.search(pattern, file):
            pattern = match.group(1)
            logsize = match.group(2)
            approach = match.group(3)
            winsize = match.group(4)
        else:
            print(f'Filename {file} do not follow the expected patter {pattern} - EXITING...')
            return

        detected_drifts, detected_delays = read_drifts_prodrift(complete_filename)
        logname = pattern + logsize
        configuration_drifts = approach + ' ' + winsize + ' drifts'
        configuration_delays = approach + ' ' + winsize + ' delays'
        if logname not in results.keys():
            results[logname] = {}

        results[logname][configuration_drifts] = detected_drifts
        results[logname][configuration_delays] = detected_delays
    df = pd.DataFrame(results).T
    out_filename = f'results_prodrift.xlsx'
    out_complete_filename = os.path.join(filepath, out_filename)
    print(f'*****************************************************************')
    print(f'Saving results at file {out_complete_filename}...')
    df.to_excel(out_complete_filename)
    print(f'*****************************************************************')


if __name__ == '__main__':
    results_filepath = 'C://Users//denisesato//OneDrive//Documents//Doutorado//Tese//experiments//Apromore//dataset1'
    file_type = '.txt'
    key = 'log_'
    filenames = get_Apromore_files(results_filepath, key, file_type)
    compile_results_from_prodrift(results_filepath, filenames)
