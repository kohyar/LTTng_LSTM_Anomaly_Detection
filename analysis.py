##########################################################################
## Analysing the detection results                                      ##
## Iman Kohyarnejadfard                                                 ##
## Polytechnique Montréal                                               ##
## analysis.py                                                          ##
##########################################################################


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import random
import os
import json
from datetime import datetime
import pandas as pd
import pytz
from colorama import Fore
from PrepareDataset import PrepareDataset
import config as config
import pandas as pd
from PrepareDataset import PrepareDataset


# Make all plotes similar
plt.figure(figsize=(40, 20))
plt.subplots_adjust(left=0.2)
plt.xlim(0, 0.5)
plt.clf()
plt.cla()


def load_data(file_):
    time_series = []
    with open(file_) as f:
        spans = json.load(f)
    for index, key in enumerate(spans.keys()):
        time_series.append(spans[key])
    print(Fore.GREEN + 'File ' + Fore.MAGENTA +
          file_ + Fore.GREEN + ' was loaded. ')
    return time_series


def make_dataset(sequence_path, sequence_complete_path, remove_new_features):
    data_set_folder = os.path.dirname(os.path.realpath(__file__)) + '/data_set'

    time_series = load_data(sequence_path)
    time_series_complete = load_data(sequence_complete_path)

    features_path = '{}/features_windowsize={}.json'.format(
        data_set_folder, config.window_size)
    with open(features_path) as f:
        initial_features = json.load(f)

    db = PrepareDataset(initial_features)
    db.process_dataset(
        {'train_set': [], 'test_set': time_series}, config.fields)

    new_features = {}
    if remove_new_features:
        test_series = []
        Numeric_test_series = []
        for index, seri in enumerate(db.test_series):
            tag = True
            for event in seri:
                if event not in db.original_features:
                    tag = False
                    break
            if tag:
                test_series.append(time_series_complete[index])
                Numeric_test_series.append(db.test_series_[index])
    else:
        for seri in db.test_series:
            for event in seri:
                if event not in db.original_features:
                    if event not in new_features.keys():
                        new_features[event] = 1
                    else:
                        new_features[event] += 1

        test_series = time_series_complete
        Numeric_test_series = db.test_series_
        for index_i, seri in enumerate(db.test_series):
            for index_j, event in enumerate(seri):
                if event not in db.original_features:
                    Numeric_test_series[index_i][index_j] = len(
                        db.original_features)

    print(Fore.GREEN + 'Number of spans: ' + Fore.MAGENTA +
          str(len(Numeric_test_series)))

    print(Fore.GREEN + 'Number of classes: ' +
          Fore.MAGENTA + str(len(db.original_features)))
    return Numeric_test_series, test_series, db.original_features, new_features


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)[0]


def format_time(timestamp):
    tz = pytz.timezone('America/Montreal')
    return pd.Timestamp(timestamp, unit='us', tz=tz)


def bar_plt(X, Y, path, y_max=20000, x_label='', y_label='', title='', title_font_size=25):
    plt.figure(figsize=(40, 20))
    plt.subplots_adjust(left=0.2)
    plt.ylim(0, y_max)
    plt.plot(X, Y)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(x_label, fontsize=25)
    plt.ylabel(y_label, fontsize=25)
    plt.title(title, fontsize=title_font_size)
    plt.savefig(path)
    plt.clf()
    plt.cla()


def freq_barh_plt(X, Y, path, x_max=0.5, x_label='', y_label='', title='', title_font_size=25):
    plt.figure(figsize=(40, 20))
    plt.subplots_adjust(left=0.2)
    plt.xlim(0, x_max)
    plt.barh(X, Y)
    plt.xticks(fontsize=20)
    plt.title(title, fontsize=title_font_size)
    plt.savefig(path)
    plt.clf()
    plt.cla()


def get_all_prediction_timestamps(window_size, time_series_complete):
    predictions_timestamps = []
    for serie_index, serie in enumerate(time_series_complete):
        for i in range(len(serie) - config.window_size):
            predicted_event = serie[i + config.window_size]
            predictions_timestamps.append(predicted_event['timestamp'])
    return predictions_timestamps


def analyse_test(test_series, time_series_complete, initial_features, json_path, name, trace_name, new_features, window_size, complete_predictions_path):
    result_folder = os.path.dirname(os.path.realpath(
        __file__))+'/analysis/{}/'.format(trace_name)
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)

    # Load mispredictions
    with open(json_path) as f:
        mispredictions = json.load(f)
    print('Number of incorrectly predicted events: {}'.format(len(mispredictions)))

    # Obtain itervals of size 1 sec and ontain the center of each interval as timestamp label of that interval
    timestamps = [event['ts'] for event in mispredictions]
    start_ts = min(timestamps)
    end_ts = max(timestamps)
    num_intervals = int((end_ts - start_ts)/(1000*1000))+1
    start_lebel = start_ts + int(interval_size/2)
    x_range = [(start_lebel + interval_size*i) for i in range(num_intervals)]

    ########################################################
    ######### Investigate misprediction over time ##########
    ########################################################
    indervals_num_misspredictions = [0 for i in range(num_intervals)]
    indervals_num_known_misspredictions = [0 for i in range(num_intervals)]
    indervals_num_unknown_misspredictions = [0 for i in range(num_intervals)]
    for event in mispredictions:
        index = int((event['ts']-start_ts)/interval_size)
        indervals_num_misspredictions[index] += 1
        if event['args']['isunknown']:
            indervals_num_unknown_misspredictions[index] += 1
        else:
            indervals_num_known_misspredictions[index] += 1

    print(Fore.YELLOW +
          'Top 20 intervals in which more errors have accoured:' + Fore.RESET)
    largest_ind = largest_indices(np.array(indervals_num_misspredictions), 20)
    text_file_path = os.path.dirname(os.path.realpath(
        __file__))+'/analysis/'+trace_name + '/' + trace_name+'_' + name+'misprediction_distribution_over_time.txt'
    f = open(text_file_path, "w")
    f.write('Distribution of incorrectly predictions over time (Trace: {}, Type: {}events)\n'.format(
        trace_name, name))
    f.write('-'*20+'\n')

    for i in largest_ind:
        ts_s = format_time(start_ts+(i*interval_size)).time()
        ts_e = format_time(start_ts+((i+1)*interval_size)).time()
        f.write('{}-{}: {}\n'.format(ts_s, ts_e,
                                     str(indervals_num_misspredictions[i])))
    f.close()

    x_range = [str(format_time((start_lebel + interval_size*i)).time())
               for i in range(num_intervals)]
    title = 'Distribution of incorrectly predictions over time (Trace: {}, Type: {}events)'.format(
        trace_name, name)
    plt_path = os.path.dirname(os.path.realpath(
        __file__))+'/analysis/'+trace_name + '/'+trace_name+'_' + name+'misprediction_distribution_over_time.png'
    bar_plt(x_range, indervals_num_misspredictions, plt_path, x_label='Timestamp',
            y_label='Number of incorrectly predicted events', title=title)

    ##########################################################
    ############ misprediction/prediction over time ##########
    ##########################################################
    all_predictions_timestamps = get_all_prediction_timestamps(
        window_size, time_series_complete)
    indervals_num_predictions = [0 for i in range(num_intervals)]
    for timestamp in all_predictions_timestamps:
        timestamp = round(timestamp/1000)
        index = int((timestamp-start_ts)/interval_size)
        if (index - len(indervals_num_predictions)) >= 0:
            for i in range((index - len(indervals_num_predictions))+1):
                indervals_num_predictions.append(0)
                indervals_num_misspredictions.append(0)
                indervals_num_known_misspredictions.append(0)
                indervals_num_unknown_misspredictions.append(0)
                x_range.append(
                    str(format_time((start_lebel + interval_size*index)).time()))
        indervals_num_predictions[index] += 1

    title = 'Number of predictions in each interval over time (Trace: {}, Type: {}events)'.format(
        trace_name, name)
    plt_path = os.path.dirname(os.path.realpath(
        __file__))+'/analysis/'+trace_name + '/'+trace_name+'_' + name+'prediction_distribution_over_time.png'
    bar_plt(x_range, indervals_num_predictions, plt_path, x_label='Timestamp',
            y_label='Number of predicted events', title=title)

    accuracy = 1 - sum(indervals_num_misspredictions) / \
        sum(indervals_num_predictions)
    indervals_misprediction_prediction = []
    for index in range(len(indervals_num_misspredictions)):
        if indervals_num_predictions[index] == 0:
            indervals_misprediction_prediction.append(0)
        else:
            indervals_misprediction_prediction.append(
                indervals_num_misspredictions[index]/indervals_num_predictions[index])

    title = 'Misprediction/Prediction in each interval over time (Trace: {}, Type: {}events, Accuracy: {}, Number of mispredictions: {}, Number of predictions: {}, num_candidates: {})'.format(
        trace_name, name, accuracy, sum(indervals_num_misspredictions), sum(indervals_num_predictions), config.num_candidates)
    plt_path = os.path.dirname(os.path.realpath(
        __file__))+'/analysis/'+trace_name + '/'+trace_name+'_' + name+'misprediction_prediction_over_time.png'
    bar_plt(x_range, indervals_misprediction_prediction, plt_path, y_max=1, x_label='Timestamp',
            y_label='Mispredictions/Predictions', title=title, title_font_size=10)

    if name == 'with_unknown_':
        indervals_known_misprediction_prediction = []
        for index in range(len(indervals_num_known_misspredictions)):
            if indervals_num_predictions[index] == 0:
                indervals_known_misprediction_prediction.append(0)
            else:
                indervals_known_misprediction_prediction.append(
                    indervals_num_known_misspredictions[index]/indervals_num_predictions[index])
        title = 'Known Misprediction/Prediction in each interval over time (Trace: {}, Type: {}events)'.format(
            trace_name, name)
        plt_path = os.path.dirname(os.path.realpath(
            __file__))+'/analysis/'+trace_name + '/'+trace_name+'_' + name+'known_misprediction_prediction_over_time.png'
        bar_plt(x_range, indervals_known_misprediction_prediction, plt_path, y_max=1, x_label='Timestamp',
                y_label='Known Mispredictions/Predictions', title=title)

        indervals_unknown_misprediction_prediction = []
        for index in range(len(indervals_num_unknown_misspredictions)):
            if indervals_num_predictions[index] == 0:
                indervals_unknown_misprediction_prediction.append(0)
            else:
                indervals_unknown_misprediction_prediction.append(
                    indervals_num_unknown_misspredictions[index]/indervals_num_predictions[index])
        title = 'Unnown Misprediction/Prediction in each interval over time (Trace: {}, Type: {}events)'.format(
            trace_name, name)
        plt_path = os.path.dirname(os.path.realpath(
            __file__))+'/analysis/'+trace_name + '/'+trace_name+'_' + name+'unknown_misprediction_prediction_over_time.png'
        bar_plt(x_range, indervals_unknown_misprediction_prediction, plt_path, y_max=1, x_label='Timestamp',
                y_label='Unknown Mispredictions/Predictions', title=title)

        plt_path = os.path.dirname(os.path.realpath(
            __file__))+'/analysis/'+trace_name + '/'+trace_name+'_' + name+'known&unknown_misprediction_prediction_over_time.png'
        title = 'Known&Unknown Misprediction/Prediction in each interval over time (Trace: {}, Type: {}events, known_misprediction_rate = {}, unknown_misprediction_rate = {})'.format(
            trace_name, name, sum(indervals_num_known_misspredictions)/sum(indervals_num_predictions), sum(indervals_num_unknown_misspredictions)/sum(indervals_num_predictions))
        plt.figure(figsize=(40, 20))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.subplots_adjust(left=0.2)
        plt.ylim(0, 1)
        plt.plot(x_range, indervals_known_misprediction_prediction,
                 label="known mispredictions")
        plt.plot(x_range, indervals_unknown_misprediction_prediction,
                 label="unknown mispredictions")
        plt.xlabel('Timestamp', fontsize=25)
        plt.ylabel('Mispredictions/Predictions', fontsize=25)
        plt.title(title, fontsize=8)
        plt.legend()
        plt.savefig(plt_path)
        plt.clf()
        plt.cla()

    ##########################################################
    ############## unknown keys ##############################
    ##########################################################
    cnt = 0
    unknowns = []
    if name == 'with_unknown_':
        #### Distribution of unknown event keys  #############
        indervals_num_unknowns = [0 for i in range(len(x_range))]
        for seq_index, seq in enumerate(test_series):
            for event_index, key in enumerate(seq):
                if key == len(initial_features):
                    unknowns.append(
                        time_series_complete[seq_index][event_index])
                    timestamp = round(
                        time_series_complete[seq_index][event_index]['timestamp']/1000)
                    index = int((timestamp-start_ts)/interval_size)
                    indervals_num_unknowns[index] += 1
                    cnt += 1
        # indervals_num_unknowns = [item/cnt for item in indervals_num_unknowns]
        title = 'Distribution of unknown keys over time (Trace: {}, Type: {}events)'.format(
            trace_name, name)
        plt_path = os.path.dirname(os.path.realpath(
            __file__))+'/analysis/'+trace_name + '/'+trace_name+'_unknown_keys_distribution_over_time.png'
        bar_plt(x_range, indervals_num_unknowns, plt_path, x_label='Timestamp',
                y_label='Number of unknown keys', title=title)

        #### Frequency of unknown event keys  ################
        sum_values = sum(new_features.values())
        unknown_keys = [key for key in new_features.keys()]
        unknown_freqs = [new_features[key] /
                         sum_values for key in new_features.keys()]
        largest_ind = largest_indices(np.array(unknown_freqs), 40)
        unknown_keys = [unknown_keys[index] for index in largest_ind]
        unknown_freqs = [unknown_freqs[index] for index in largest_ind]
        plt_path = os.path.dirname(os.path.realpath(
            __file__))+'/analysis/'+trace_name + '/'+trace_name+'_unknown_keys_frequency.png'
        title = 'Frequency of unknown events in test set (Trace: {}, Type: {}events)'.format(
            trace_name, name)
        freq_barh_plt(unknown_keys, unknown_freqs, plt_path, title=title)

    ##########################################################
    ############## Frequency of features in test set   #######
    ##########################################################
    key_freqs = [0 for item in initial_features]
    keys = [item for item in initial_features]
    num_events = 0
    if name == 'with_unknown_':
        key_freqs.append(0)
        keys.append('unknown')
    for seq in test_series:
        for index in seq:
            key_freqs[index] += 1
            num_events += 1
    key_freqs = [item/num_events for item in key_freqs]
    ordered_inds = largest_indices(
        np.array(key_freqs), 30)

    ordered_key_freqs = []
    ordered_keys = []
    for index in ordered_inds:
        ordered_key_freqs.append(key_freqs[index])
        ordered_keys.append(keys[index])

    # Make plot
    plt_path = os.path.dirname(os.path.realpath(
        __file__))+'/analysis/' + trace_name + '/'+trace_name+'_' + name+'frequency_of_keys_testset.png'
    title = 'Frequency of keys in test set (Trace: {}, Type: {}events)'.format(
        trace_name, name)
    freq_barh_plt(ordered_keys, ordered_key_freqs, plt_path, title=title)

    ##########################################################

    ##########################################################
    ############# Distribution of mispredictions##############
    ##########################################################
    misprediction_key_distribution = [0 for item in initial_features]
    if name == 'with_unknown_':
        misprediction_key_distribution.append(0)

    for event in mispredictions:
        index = event['args']['keyIndex']
        misprediction_key_distribution[index] += 1
    misprediction_key_distribution = [
        item/len(mispredictions) for item in misprediction_key_distribution]
    ordered_inds = largest_indices(
        np.array(misprediction_key_distribution), 30)

    ordered_misprediction_key_distribution = []
    ordered_misprediction_keys = []
    for index in ordered_inds:
        ordered_misprediction_key_distribution.append(
            misprediction_key_distribution[index])
        ordered_misprediction_keys.append(keys[index])

    # Make plot
    title = 'Frequency of incorrectly predicted keys in test data (Trace: {}, Type: {}events)'.format(
        trace_name, name)
    plt_path = os.path.dirname(os.path.realpath(
        __file__))+'/analysis/'+trace_name + '/'+trace_name+'_'+name+'frequency_of_incorrectly_predicted_test.png'
    freq_barh_plt(ordered_misprediction_keys,
                  ordered_misprediction_key_distribution, plt_path, title=title)
    ##########################################################

    return x_range, indervals_misprediction_prediction


def service_base_analysis(complete_predictions_path, trace_name):
    if not os.path.isdir(os.path.dirname(os.path.realpath(
        __file__))+'/analysis/' + trace_name + '/service_based/'):
        os.makedirs(os.path.dirname(os.path.realpath(
        __file__))+'/analysis/' + trace_name + '/service_based/')
    colors = ['lime', 'darkgreen', 'pink', 'indigo', 'black']
    # colors = ['lime', 'green', 'cyan', 'dodgerblue', 'lightpink','mediumvioletred','blueviolet','darkslateblue', 'navy']
    print((Fore.YELLOW + '-'*15+Fore.GREEN + '-' *
           15+Fore.MAGENTA + '-'*15)*3 + Fore.RESET)
    with open(complete_predictions_path) as f:
        predicted_sequences = json.load(f)

    spans_num_misprediction = []
    for index, sequence in enumerate(predicted_sequences):
        spans_num_misprediction.append(0)
        for event in sequence:
            if event.startswith('$'):
                spans_num_misprediction[index] += 1
    spans_num_predictions = [0 if (len(seq)-config.window_size) < 0 else (
        len(seq)-config.window_size) for seq in predicted_sequences]
    prob_mispredictions = [0 if spans_num_predictions[index] == 0 else spans_num_misprediction[index] /
                           spans_num_predictions[index] for index, seq in enumerate(predicted_sequences)]

    # PLOT 0 : x_axis: spans length, y_axis: number of mispredictions
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    X_cordinates = [len(predicted_sequences[index]) for index, item in enumerate(predicted_sequences)]
    Y_cordinates = [spans_num_misprediction[index] for index, item in enumerate(predicted_sequences)]
    point_colors = ['red' for index, item in enumerate(predicted_sequences)]
    plt.scatter(X_cordinates, Y_cordinates, color= point_colors, s=100)
    plt.xlabel('Spans length', fontsize=25)
    plt.ylabel('Number of mispredictions',  fontsize=25)
    plt_path = os.path.dirname(os.path.realpath(
        __file__))+'/analysis/' + trace_name + '/service_based/'+trace_name + '_point_spanlength_mspredictions'
    plt.savefig(plt_path)
    plt.clf()
    plt.cla()

    # PLOT 1 : x_axis: Spans index, y_axis: Number of mispredictions/Number of predictions, color: prob(misprediction)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    X_cordinates = [index for index, item in enumerate(predicted_sequences)]
    Y_cordinates = [prob_mispredictions[index] for index, item in enumerate(predicted_sequences)]
    point_colors = [colors[int(prob_mispredictions[index]/0.25)] for index, item in enumerate(predicted_sequences)]
    plt.scatter(X_cordinates, Y_cordinates, color=point_colors, s=10)
    plt.xlabel('Spans index', fontsize=25)
    plt.ylabel('Number of mispredictions/Number of predictions',  fontsize=25)
    xlim = len(X_cordinates)
    plt.ylim(0, 1)
    plt.xlim(0, xlim)
    plt_path = os.path.dirname(os.path.realpath(
        __file__))+'/analysis/' + trace_name + '/service_based/'+trace_name+'_point_spanindex_mispredictionProb.png'
    plt.savefig(plt_path)
    plt.clf()
    plt.cla()

    # PLOT 2 : x_axis: Spans index, y_axis: Number of mispredictions/Number of predictions, color: prob(misprediction)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    X_cordinates = [index for index, item in enumerate(predicted_sequences) if prob_mispredictions[index] >= 0.5]
    Y_cordinates = [prob_mispredictions[index] for index, item in enumerate(predicted_sequences) if prob_mispredictions[index] >= 0.5]
    point_colors = [colors[int(prob_mispredictions[index]/0.25)] for index, item in enumerate(predicted_sequences) if prob_mispredictions[index] >= 0.5]
    plt.scatter(X_cordinates, Y_cordinates, color=point_colors, s=10)
    plt.xlabel('Spans index', fontsize=25)
    plt.ylim(0, 1)
    plt.xlim(0, xlim)
    plt.ylabel('Number of mispredictions/Number of predictions',  fontsize=25)
    plt_path = os.path.dirname(os.path.realpath(
        __file__))+'/analysis/' + trace_name + '/service_based/'+trace_name+'_point_spanindex_misprediction_HighProb.png'
    plt.savefig(plt_path)
    plt.clf()
    plt.cla()

    ### PLOT 3 : x_axis: Spans index, y_axis: Number of predictions, color: prob(misprediction)
    ylim = max(spans_num_predictions)
    xlim = len(spans_num_predictions)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    for index, item in enumerate(predicted_sequences):
        color = colors[int(prob_mispredictions[index]/0.25)]
        plt.bar(index , spans_num_predictions[index] , color = color)
    plt.xlabel('Spans index', fontsize=25)
    plt.ylabel('Number of predictions',  fontsize=25)
    plt_path = os.path.dirname(os.path.realpath(
        __file__))+'/analysis/' + trace_name + '/service_based/'+trace_name+'_bar_spanindex_numPrediction_colored.png'
    plt.xlim(0, xlim)
    plt.ylim(0, ylim)
    plt.savefig(plt_path)
    plt.clf()
    plt.cla()


    ### PLOT 4 : x_axis: Spans index, y_axis: Number of predictions, color: prob(misprediction)
    ylim = max(spans_num_predictions)
    xlim = len(spans_num_predictions)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    for index, item in enumerate(predicted_sequences):
        if prob_mispredictions[index] >= 0.5:
            color = colors[int(prob_mispredictions[index]/0.25)]
            plt.bar(index , spans_num_predictions[index] , color = color)
    plt.xlabel('Spans index', fontsize=25)
    plt.ylabel('Number of predictions',  fontsize=25)
    plt_path = os.path.dirname(os.path.realpath(
        __file__))+'/analysis/' + trace_name + '/service_based/'+trace_name+'_bar_spanindex_numPrediction_HighProb_colored_.png'
    plt.xlim(0, xlim)
    plt.ylim(0, ylim)
    plt.savefig(plt_path)
    plt.clf()
    plt.cla()


    # PLOT 5 : x_axis: Spans index, y_axis: Number of predictions, color: prob(misprediction)
    ylim = max(spans_num_predictions)
    number_of_subplots = int(len(predicted_sequences)/500)+1
    temp_predicted_sequences = [[] for index in range(number_of_subplots*500)]
    for index, seq in enumerate(predicted_sequences):
        temp_predicted_sequences[index] = predicted_sequences[index]
    
    temp_prob_mispredictions = [0 for index in range(number_of_subplots*500)]
    for index, seq in enumerate(prob_mispredictions):
        temp_prob_mispredictions[index] = prob_mispredictions[index]

    fig, axes = plt.subplots(number_of_subplots, figsize=(30, 20))
    for subplot_index in range(number_of_subplots):
        for index in range(subplot_index*500, (subplot_index+1)*500):
            color = colors[int(temp_prob_mispredictions[index]/0.25)]
            axes[subplot_index].bar(index, len(
                temp_predicted_sequences[index])-config.window_size, color=color)
            axes[subplot_index].set_xlim(subplot_index*500, (subplot_index+1)*500)
            axes[subplot_index].set_ylim(0, ylim)
            axes[subplot_index].set_xlabel('Spans index')
            axes[subplot_index].set_ylabel('Number of predictions')
            # axes[subplot_index].axhline(y=0,color='gray')

    plt_path = os.path.dirname(os.path.realpath(
        __file__))+'/analysis/' + trace_name + '/service_based/'+trace_name+'_multiBar_spanindex_numPrediction_colored.png'
    plt.savefig(plt_path)
    plt.clf()
    plt.cla()


    # PLOT 6 : x_axis: Spans index, y_axis: Number of predictions, color: prob(misprediction)
    ylim = max(spans_num_predictions)
    fig, axes = plt.subplots(number_of_subplots, figsize=(30, 20))
    for subplot_index in range(number_of_subplots):
        for index in range(subplot_index*500, (subplot_index+1)*500):
            if temp_prob_mispredictions[index] >= 0.5:
                color = colors[int(temp_prob_mispredictions[index]/0.25)]
                axes[subplot_index].bar(index, len(
                    temp_predicted_sequences[index])-config.window_size, color=color)
                axes[subplot_index].set_xlim(subplot_index*500, (subplot_index+1)*500)
                axes[subplot_index].set_ylim(0, ylim)
                axes[subplot_index].set_xlabel('Spans index')
                axes[subplot_index].set_ylabel('Number of predictions')
                # axes[subplot_index].axhline(y=0,color='gray')

    plt_path = os.path.dirname(os.path.realpath(
        __file__))+'/analysis/' + trace_name + '/service_based/'+trace_name+'_multiBar_spanindex_numPrediction_HighProb_colored.png'
    plt.savefig(plt_path)
    plt.clf()
    plt.cla()

    # PLOT 8 : x_axis: Spans index, y_axis: Number of predictions, color: red
    high_prob_spans_num_predictions = []
    for index, prob in enumerate(prob_mispredictions):
        if prob >= 0.5:
            high_prob_spans_num_predictions.append(spans_num_predictions[index])
    ylim = max(high_prob_spans_num_predictions)+500
    
    fig, axes = plt.subplots(number_of_subplots, figsize=(30, 20))
    for subplot_index in range(number_of_subplots):
        axes[subplot_index].tick_params(axis='both', which='major', labelsize=20)
        for index in range(subplot_index*500, (subplot_index+1)*500):
            if temp_prob_mispredictions[index] >= 0.5:
                color = 'red'
                if len(temp_predicted_sequences[index])-config.window_size <100:
                    axes[subplot_index].bar(index, len(
                        temp_predicted_sequences[index])-config.window_size+100, color=color)
                else:    
                    axes[subplot_index].bar(index, len(
                        temp_predicted_sequences[index])-config.window_size, color=color)
                axes[subplot_index].set_xlim(subplot_index*500, (subplot_index+1)*500)
                axes[subplot_index].set_ylim(0, ylim)
                # axes[subplot_index].set_xlabel('Spans index', fontsize=25)
                # axes[subplot_index].set_ylabel('Number of predictions', fontsize=25)
                # axes[subplot_index].axhline(y=0,color='gray')

    plt_path = os.path.dirname(os.path.realpath(
        __file__))+'/analysis/' + trace_name + '/service_based/'+trace_name+'_multiBar_spanindex_numPrediction_HighProb_red.png'
    plt.savefig(plt_path)
    plt.clf()
    plt.cla()


    # PLOT 7 : x_axis: Spans index, y_axis: Number of known & unknown predictions, color: prob(misprediction)
    temp_predicted_sequences = [[] for index in range(number_of_subplots*500)]
    for index, seq in enumerate(predicted_sequences):
        temp_predicted_sequences[index] = predicted_sequences[index]
    
    num_known_predictions, num_unknown_predictions = [], []
    for index, seq in enumerate(temp_predicted_sequences):
        num_known_predictions.append(0)
        num_unknown_predictions.append(0)
        for event in seq:
            if event.startswith('$'):
                if event.startswith('$#'):
                    num_unknown_predictions[index] += 1
                else:
                    num_known_predictions[index] += 1

    ylim = max(max(num_known_predictions), max(num_unknown_predictions))
    fig, axes = plt.subplots(number_of_subplots, figsize=(30, 20))
    for subplot_index in range(number_of_subplots):
        for index in range(subplot_index*500, (subplot_index+1)*500):
            if temp_prob_mispredictions[index] >= 0.5:
                color = colors[int(temp_prob_mispredictions[index]/0.25)]
                axes[subplot_index].bar(index, num_known_predictions[index], color=color)
                axes[subplot_index].bar(index, num_unknown_predictions[index]*-1, color=color)
                axes[subplot_index].set_xlim(subplot_index*500, (subplot_index+1)*500)
                axes[subplot_index].set_ylim(ylim*-1, ylim)
                axes[subplot_index].set_xlabel('Spans index')
                axes[subplot_index].axhline(y=0,color='gray', linewidth=0.01)

    plt_path = os.path.dirname(os.path.realpath(
        __file__))+'/analysis/' + trace_name + '/service_based/'+trace_name+'_multiBar_spanindex_numKnown&UnknownPrediction_HighProb_colored.png'
    plt.savefig(plt_path)
    plt.clf()
    plt.cla()
    print()


if __name__ == '__main__':
    window_size = config.window_size
    interval_size = 1000*1000
    if not os.path.isdir(os.path.dirname(os.path.realpath(__file__))+'/analysis/'):
        os.makedirs(os.path.dirname(os.path.realpath(__file__))+'/analysis/')
    sequence_path = '' #To be set
    sequence_complete_path = '' #To be set
    complete_predictions_path = ''#To be set
    trace_name = '' #To be set
    
    print(Fore.YELLOW + 'Step 1: ' + Fore.MAGENTA + 'Service Perspective' + Fore.RESET)
    service_base_analysis(complete_predictions_path, trace_name)
    print(Fore.YELLOW + 'Step 1: ' + Fore.MAGENTA + 'Completed' + Fore.RESET)
    print(Fore.YELLOW + 'Step 2: ' + Fore.MAGENTA + 'Analyse the results' + Fore.RESET)
    google_json_format_path = '{}/google_json_format_output/{}_test_result_with_unknown.json'.format(
        os.path.dirname(os.path.realpath(__file__)), trace_name)
    test_series, time_series_complete, initial_features, new_features = make_dataset(
        sequence_path, sequence_complete_path, False)
    analyse_test(test_series, time_series_complete, initial_features,google_json_format_path, '', trace_name, new_features, window_size, complete_predictions_path)
    print(Fore.YELLOW + 'Step 2: ' + Fore.MAGENTA + 'Completed' + Fore.RESET)

        