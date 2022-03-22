##########################################################################
## Read events obtained from traces and make sequences of keys          ##
## Iman Kohyarnejadfard                                                 ##
## Polytechnique Montréal                                               ##
## process_events.py                                                    ##
##########################################################################

from load_ReqReps import load_ReqReps
import csv
import json
from colorama import Fore
import os
import numpy as np
import config as config


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

    