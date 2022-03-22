##########################################################################
## Prepare the dataset                                                  ##
## Iman Kohyarnejadfard                                                 ##
## Polytechnique Montréal                                               ##
## PrepareDataset.py                                                    ##
##########################################################################

from colorama import Fore
import os
import json
from random import randint
import gc


class PrepareDataset:
    def __init__(self, features=[], noise=[]):
        self.features = features
        self.original_features = features.copy()
        self.series = []
        self.test_series = []
        self.series_ = []
        self.test_series_ = []
        self.noise = noise

        self.series_index = []
        self.X = []
        self.Y = []

    def remove_bad_time_series(self, sequences):
        series = []
        for index, time_serie in enumerate(sequences):
            status = True
            if len(time_serie) < 4:
                status = False
            elif time_serie[0].startswith('RESP'):
                status = False
            elif not time_serie[len(time_serie)-1].startswith('RESP*/0*'):
                status = False
            if status == True:
                series.append(time_serie)
        return series


    def prepare_keys(self, fields, series):
        temp_series = []
        for i, serie in enumerate(series):
            temp_serie = []
            for j, event in enumerate(serie):
                event_temp = event
                events_split = event.split('*')
                if events_split[0].startswith('REQ') or events_split[0].startswith('RES'):
                    event_temp = events_split[0]
                    if fields['msgTrace:reqrep']['alias']:
                        event_temp = event_temp + '*' + events_split[1]
                    if fields['msgTrace:reqrep']['procname']:
                        event_temp = event_temp + '*' + events_split[2]

                elif 'ddmZmq:OAL_LOG_DEBUG' == events_split[0]:
                    event_temp = events_split[0]
                    if fields['ddmZmq:OAL_LOG_DEBUG']['procname']:
                        event_temp = event_temp + '*' + events_split[1]
                    if fields['ddmZmq:OAL_LOG_DEBUG']['msg']:
                        event_temp = event_temp + '*' + events_split[2]

                elif 'oal:OAL_LOG_DEBUG' == events_split[0]:
                    event_temp = events_split[0]
                    if fields['oal:OAL_LOG_DEBUG']['procname']:
                        event_temp = event_temp + '*' + events_split[1]
                    if fields['oal:OAL_LOG_DEBUG']['func']:
                        event_temp = event_temp + '*' + events_split[2]
                else:
                    event_temp = events_split[0]
                    if fields['all']['procname']:
                        event_temp = event_temp + '*' + events_split[1]

                temp_serie.append(event_temp)
            temp_series.append(temp_serie)
        return temp_series

    def find_all_features(self):
        for serie in self.series:
            for event in serie:
                if event not in self.features:
                    self.features.append(event)
        for serie in self.test_series:
            for event in serie:
                if event not in self.features:
                    self.features.append(event)

    def to_numeric(self):
        for serie in self.series:
            numeric_serie = []
            for event in serie:
                numeric_serie.append(self.features.index(event))
            self.series_.append(numeric_serie)
        for serie in self.test_series:
            numeric_serie = []
            for event in serie:
                numeric_serie.append(self.features.index(event))
            self.test_series_.append(numeric_serie)

    def get_train_sequences(self, seq_len):
        for serie in self.series_:
            if len(serie) > (seq_len+1):
                n_sub_series = len(serie)-seq_len
                for i in range(1, n_sub_series+1):
                    x = serie[i-1:seq_len+i-1]
                    y = serie[seq_len+i-1]
                    self.X.append(x)
                    self.Y.append(y)
        print(Fore.GREEN + 'All sequences of size ' + Fore.YELLOW +
              str(seq_len) + Fore.GREEN + ' were extracted.')
        return self.X, self.Y

    def get_test_sequences(self, seq_len):
        indexes = []
        for index, serie in enumerate(self.test_series_):
            if len(serie) > (seq_len+1):
                n_sub_series = len(serie)-seq_len
                for i in range(1, n_sub_series+1):
                    x = serie[i-1:seq_len+i-1]
                    y = serie[seq_len+i-1]
                    self.X.append(x)
                    self.Y.append(y)
                    indexes.append(index)
        print(Fore.GREEN + 'All sequences of size ' + Fore.YELLOW +
              str(seq_len) + Fore.GREEN + ' were extracted.')
        return self.X, self.Y, indexes

    def process_dataset(self, data, fields):
        print(Fore.GREEN + 'Preparing sequences...')
        series = data['train_set']
        test_series = data['test_set']
        self.test_series_complete = test_series
        self.series = self.prepare_keys(fields, series)
        self.test_series = self.prepare_keys(fields, test_series)
        self.find_all_features()
        if len(self.noise) > 0:
            self.series = self.series + self.noise
            self.features.append('unknown')
        self.to_numeric()
