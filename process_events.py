##########################################################################
## Read events obtained from traces and make sequences of keys and      ##
## train model                                                          ##
## Iman Kohyarnejadfard                                                 ##
## Polytechnique Montréal                                               ##
## process_events.py                                                    ##
##########################################################################

from load_ReqReps import load_ReqReps
import config as config
from PrepareDataset import PrepareDataset
import csv
import json
from colorama import Fore
import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import argparse
from sklearn.model_selection import train_test_split
import datetime
from random import randint
import psutil
import torch.cuda as cutorch
from torch.utils.data import Dataset
import random
import copy


# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
print(Fore.GREEN + 'Detected Device: ' +
      Fore.MAGENTA + str(device) + Fore.RESET)

def load_ust_events(ust_path):
    rows = []
    with open(ust_path) as myFile:
        reader = csv.DictReader(myFile)
        for row in reader:
            if row['name']!='Lost event':
                ust_event = {}
                ust_event['name'] = row['name']
                ust_event['timestamp'] = float(row['timestamp'])
                ust_event['msgTag'] = row['msgTag']
                ust_event['vtid'] = int(row['vtid'])
                ust_event['vpid'] = int(row['vpid'])
                ust_event['procname'] = row['procname']
                ust_event['msgType'] = row['msgType']
                ust_event['pl'] = row['pl']
                ust_event['func'] = row['func']
                ust_event['msg'] = row['msg']
                rows.append(ust_event)
    return rows


def make_spans(ust_events):
    lrr = load_ReqReps(ust_events)
    lrr.make_spans()
    spans, ust_events = lrr.get_spans()
    return spans, ust_events


# This function return a specific part of the event_tag
def find_tag(event_tag, index):
    return event_tag.split('"tag":')[index + 1].split('}')[0]

def find_real_tag(msg_tag):
    sub_tags = [(find_tag(msg_tag, i) + '/')
                for i in range(msg_tag.count('"tag":'))]
    real_tag = ''
    for item in sub_tags:
        real_tag+=item
    return real_tag


def find_alias(msg_tag, spans):
    alias = ''
    sub_tags = [(find_tag(msg_tag, i) + '/')
                for i in range(msg_tag.count('"tag":'))]
    if len(sub_tags) == 1:
        try:
            tag = spans[sub_tags[0]]['aliases'][sub_tags[0]]
        except:
            spans[sub_tags[0]]['aliases'] = {sub_tags[0]: 0}
            spans[sub_tags[0]]['max_alias'] = 0
            tag = 0
        alias = '/{}'.format(0)
    else:
        for item in sub_tags:
            if item not in spans[sub_tags[0]]['aliases'].keys():
                spans[sub_tags[0]]['max_alias'] += 1
                spans[sub_tags[0]]['aliases'][item] = spans[
                    sub_tags[0]]['max_alias']
            alias = alias + '/{}'.format(spans[sub_tags[0]]['aliases'][item])
    return alias, spans


def include_pid(span, event):
    for tidPid in span['tidPid']:
        if tidPid['pid'] == event['vpid']:
            return True
    return False


def include_tid(span, event):
    for tidPid in span['tidPid']:
        if tidPid['tid'] == event['vtid']:
            return True
    return False


def find_corresponded_span(event, active_spans, active_spans_start,
                           active_spans_end, spans):
    corresponded_span_id = ''
    for index, span_id in enumerate(active_spans):
        if event['timestamp'] >= active_spans_start[index] and event[
                'timestamp'] <= active_spans_end[index]:
            included_pid = include_pid(spans[span_id], event)
            included_tid = include_tid(spans[span_id], event)
            if included_pid and included_tid:
                corresponded_span_id = span_id
                break
    return corresponded_span_id

def oal_msg_alias(msg):
    msg_alias = ''
    if ':' in msg:
        msg = msg.split(':')[0]
    if ' ' in msg:
        msg_alias = msg.split(' ')[0] + '-' + msg.split(' ')[1]  
    elif '\\' in msg:
        msg_alias = msg.split('\\')[0]
    else:
        msg_alias=msg
    return msg_alias

def assign_ust_events(ust_events=None, all_events=None, spans=None):
    sequence_of_events = {}
    active_spans = []
    active_spans_start = []
    active_spans_end = []
    for event in ust_events:
        if event['name'] == 'msgTrace:reqrep':
            root_tag = find_tag(event['msgTag'], 0) + '/'
            real_tag = find_real_tag(event['msgTag'])
            if root_tag in spans:
                if event['status'] == True and event['timestamp'] <= (
                        spans[root_tag]['timestamp'] +
                        spans[root_tag]['duration']):
                    alias, spans = find_alias(event['msgTag'], spans)
                    if root_tag not in active_spans:
                        # Create a new sequence in sequence_of_events
                        active_spans.append(root_tag)
                        # We add an new span to active_spans
                        active_spans_start.append(spans[root_tag]['timestamp'])
                        active_spans_end.append(spans[root_tag]['timestamp'] +
                                                spans[root_tag]['duration'])
                        sequence_of_events[root_tag] = [
                            '{}*{}*{}*{}'.format(event['msgType'], alias,
                                              event['procname'], real_tag)
                        ]
                    else:
                        # We add this event to the corresponded span
                        sequence_of_events[root_tag].append('{}*{}*{}*{}'.format(
                            event['msgType'], alias, event['procname'], real_tag))
                        # Check if this is the last event in the span
                        index = active_spans.index(root_tag)
                        if event['timestamp'] == active_spans_end[index]:
                            del active_spans[index]
                            del active_spans_start[index]
                            del active_spans_end[index]
        
        elif event['name'] == 'ddmZmq:OAL_LOG_DEBUG':
            span_id = find_corresponded_span(event, active_spans,
                                             active_spans_start,
                                             active_spans_end, spans)
            if span_id != '':
                msg_start = event['msg'].split('{')[0].replace(' ','')
                if 'tcp' in msg_start:
                    msg_start = 'publishing'
                    msg_path = 'none'
                elif 'Timed-out' in msg_start:
                    msg_start = msg_start.split('Timed-out')[0]
                    msg_path = 'Timed-out'
                else:
                    msg_path = event['msg'].split('"path":')[1].split(',')[0]
                    if 'ciena' in msg_path:
                        msg_path = 'none'
                msg_start.replace('(', '').replace(')', '').replace('=', '')
                sequence_of_events[span_id].append('{}*{}*{}${}'.format(event['name'], event['procname'], msg_start, msg_path))

        elif event['name'] =='oal:OAL_LOG_DEBUG':
            span_id = find_corresponded_span(event, active_spans,
                                             active_spans_start,
                                             active_spans_end, spans)
            if span_id != '':
                if event['procname']=='ddm-zmq-worker':
                    func = event['func'].split('_')[0]
                    sequence_of_events[span_id].append('{}*{}*{}'.format(event['name'], event['procname'], func))
                else:
                    sequence_of_events[span_id].append('{}*{}*none'.format(event['name'], event['procname']))

        else:
            span_id = find_corresponded_span(event, active_spans,
                                             active_spans_start,
                                             active_spans_end, spans)
            if span_id != '':
                sequence_of_events[span_id].append('{}*{}'.format(event['name'], event['procname']))

    return sequence_of_events


def process_trace(ust_path=None, all_events_path=None):
    root_directory = os.path.dirname(ust_path)
    current_file_name = ust_path.split('/')[len(ust_path.split('/')) -
                                            1].split('.')[0]
    print(Fore.MAGENTA + 'EventsToSeq is working on {}'.format(ust_path))
    print(Fore.YELLOW + 'Step 1: Load ust events...')
    ust_events = load_ust_events(ust_path)
    print(Fore.GREEN + 'Step 1: ust events were loaded!!')

    print(Fore.YELLOW + 'Step 2: Extracting spans...')
    spans, ust_events = make_spans(ust_events)
    # Create spans.json and spans_events.json
    spans_path = '{}/spans/spans_{}.json'.format(root_directory,
                                                 current_file_name)
    if not os.path.exists('{}/spans/'.format(root_directory)):
        os.makedirs('{}/spans/'.format(root_directory))
    with open(spans_path, 'w') as json_file:
        json.dump(spans, json_file)
    print(Fore.GREEN +
          'Step 2: Spans were successfully extracted and stored in: {}'.format(
              spans_path))

    if all_events_path == None:
        print(Fore.YELLOW + 'Step 3: Assign ust events to spans...')
        sequences = assign_ust_events(ust_events=ust_events, spans=spans)
        print(Fore.GREEN +
              'Step 3: ust events were assigned to spans successfully!!')

        print(Fore.YELLOW + 'Step 4: Creat output file...')
        if not os.path.exists('{}/train_sequences/'.format(root_directory)):
            os.makedirs('{}/train_sequences/'.format(root_directory))
        out_seq_path = '{}/train_sequences/{}.json'.format(root_directory,
                                                     current_file_name)
        with open(out_seq_path, 'w') as json_file:
            json.dump(sequences, json_file)
        print(Fore.GREEN +
              'Step 4: Output file were successfully created in: {}'.format(
                  out_seq_path))

    print(Fore.MAGENTA + '-' * 10 + Fore.GREEN + '-' * 10 + Fore.YELLOW +
          '-' * 10 + Fore.MAGENTA + '-' * 10 + Fore.GREEN + '-' * 10 +
          Fore.YELLOW + '-' * 10)



def show_system_info(GPU_card_index):
    a = torch.cuda.memory_allocated(GPU_card_index)/(1024*1024)
    t = torch.cuda.get_device_properties(GPU_card_index).total_memory/(1024*1024)
    print(Fore.GREEN + 'mem: ' + Fore.MAGENTA + '{}%, '.format(psutil.virtual_memory().percent) +
        Fore.GREEN + 'CPU: ' + Fore.MAGENTA + '{}%, '.format(psutil.cpu_percent()) +
        Fore.GREEN + 'GPU-mem(MB): ' + Fore.MAGENTA + '{} of {}'.format(int(a),int(t)) +
        Fore.RESET)


# This function reads all cvs files in folder_path and obtains sequences
def load_data(folder_path):
    time_series = []
    file_list = os.listdir(folder_path)
    for file_ in file_list:
        with open('{}/{}'.format(folder_path, file_)) as f:
            spans = json.load(f)
        for key in spans.keys():
            time_series.append(spans[key])
        print(Fore.GREEN + 'File ' + Fore.MAGENTA +
              file_ + Fore.GREEN + ' was loaded. ')
    return time_series


def create_randomized_set(sequences):
    order = np.arange(len(sequences))
    random.shuffle(order)
    new_sequences = []
    for index in order:
        new_sequences.append(sequences[index])
    return new_sequences


def make_dataset(folder_path, window_size):
    data = []
    data_set_folder = os.path.dirname(os.path.realpath(__file__)) + '/data_set'
    dataset_path = '{}/dataset.json'.format(data_set_folder)
    features_path = '{}/features_windowsize={}.json'.format(data_set_folder,window_size)
    if not os.path.isfile(dataset_path):
        if not os.path.isdir(data_set_folder):
            os.makedirs(data_set_folder)
        time_series = load_data(folder_path)
        time_series = create_randomized_set(time_series)

        random_indexes = []
        for i in range(int(len(time_series)*config.test_percentage)):
            rand_ind = randint(0, len(time_series))
            if rand_ind not in random_indexes:
                random_indexes.append(rand_ind)

        train_series = []
        test_series = []
        for i in range(len(time_series)):
            if i in random_indexes:
                test_series.append(time_series[i])
            else:
                train_series.append(time_series[i])
        data = {
            'test_set': test_series,
            'train_set': train_series
        }
        with open(dataset_path, 'w') as json_file:
            json.dump(data, json_file)

        print(Fore.GREEN + 'Dataset was stored in: ' +
              Fore.MAGENTA + dataset_path + Fore.GREEN)

    else:
        print(Fore.GREEN + 'Dataset file exists.')
        with open(dataset_path) as f:
            data = json.load(f)
        print(Fore.GREEN + 'Dataset is loaded.')

    db = PrepareDataset()
    db.process_dataset(data, config.fields)
    inputs, outputs = db.get_train_sequences(window_size)
    with open(features_path, 'w') as json_file:
        json.dump(db.features, json_file)
    num_classes = len(db.features)
    train_series_ = db.series_
    test_series_ = db.test_series_
    num_sessions = len(db.series)
    num_seqs = len(db.X)
    del db

    print(Fore.GREEN + 'Number of sessions: ' +
          Fore.MAGENTA + str(num_sessions))
    print(Fore.GREEN + 'Number of seqs: ' + Fore.MAGENTA + str(num_seqs))
    print(Fore.GREEN + 'Number of classes: ' + Fore.MAGENTA + str(num_classes))

    return inputs, outputs, train_series_, test_series_, num_classes, num_sessions, num_seqs


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_keys):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class timeseries(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y)
        self.len = x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len


def train_lstm(window_size=config.window_size, model_path = config.model_path):
    # Read .csv files and extract all sequence of size k and make dataset
    print(Fore.YELLOW + 'Train-Step1: ')
    folder_path = os.path.dirname(
        os.path.realpath(__file__)) + '/train_sequences'
    inputs, outputs, train_series_, validation_series_, num_classes, num_sessions, num_seqs = make_dataset(folder_path, window_size)
    num_classes+=1
    # Train LSTM Model
    print(Fore.YELLOW + 'train-Step2: ')
    if not os.path.isfile(model_path):
        model = Model(config.input_size, config.hidden_size,
                      config.num_layers, num_classes).to(device)

        dataset = timeseries(np.array(inputs), np.array(outputs))
        dataloader = DataLoader(dataset, num_workers=8,
                                shuffle=True, batch_size=config.batch_size)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())

        # Train the model
        start_time = time.time()
        total_step = len(dataloader)
        for epoch in range(config.num_epochs):  # Loop over the dataset multiple times
            train_loss = 0
            for step, (seq, label) in enumerate(dataloader):
                # Forward pass
                seq = seq.clone().detach().view(-1, window_size, config.input_size).to(device)
                output = model(seq)
                loss = criterion(output, label.to(device))
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            print('Epoch [{}/{}], train_loss: {:.4f}'.format(epoch +
                                                             1, config.num_epochs, train_loss / total_step))

        loss = train_loss / total_step
        elapsed_time = time.time() - start_time
        print('elapsed_time: {:.3f}s'.format(elapsed_time))

        # Store Model
        print(Fore.YELLOW + 'Step3: ' + Fore.RESET)
        model_dir = os.path.dirname(os.path.realpath(__file__)) + '/model'
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        log = 'batch_size={}_epoch={}'.format(
            str(config.batch_size), str(config.num_epochs))
        torch.save(model.state_dict(), model_path)
        print(Fore.GREEN + 'Model stored in: ' +
              Fore.MAGENTA + model_path + Fore.RESET)
    else: 
        print(Fore.GREEN + 'Model already exists!!' + Fore.RESET)
        model = Model(config.input_size, config.hidden_size,
                  config.num_layers, num_classes).to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()
  

if __name__ == '__main__':
    os.system('clear')
    process_type = 'ust'  # 'ust' or 'all'
    
    # process all csv file in the folder traces_folder_path
    processed_traces_folder = config.traces_folder_path
    items = os.listdir(processed_traces_folder)
    processed_trace_list = [item for item in items if '.csv' in item]
    for item in processed_trace_list:
        file_path = processed_traces_folder + item
        if os.path.isfile(file_path):
            process_trace(ust_path=file_path)

    train_lstm()
    

    