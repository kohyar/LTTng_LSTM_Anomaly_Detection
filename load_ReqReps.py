##########################################################################
## Obtain spans                                                         ##
## Iman Kohyarnejadfard                                                 ##
## Polytechnique Montréal                                               ##
## load_ReqReps.py                                                      ##
##########################################################################

class LoadReqRepSpans:
    def __init__(self, data):
        self.trace_collection = data
        self.spans = {}

    
    # This function return a specific part of the event_tag
    def find_tag(self, event_tag, index):
        return event_tag.split('"tag":')[index+1].split('}')[0]

    # This function returns True if the father of the subspan was seen previously in the trace
    def should_consider_subspan(self, msg_tag, root_tag):
        temp_tags = []
        full_path = ''
        sub_path = ''
        should_consider = False
        for i in range(msg_tag.count('"tag":')):
            temp_tags.append(self.find_tag(msg_tag, i))
        for i in range(len(temp_tags)):
            full_path = full_path + temp_tags[i] + '/'
        for i in range(len(temp_tags)-1):
            sub_path = sub_path + temp_tags[i] + '/'

        if sub_path == root_tag and root_tag in self.spans.keys():
            should_consider = True
        elif root_tag in self.spans.keys() and sub_path in self.spans[root_tag].keys():
            should_consider = True

        return should_consider, full_path

    # This function returns True if a new combination of (tid,bid,procname) is seen for a subspan
    def should_add_tidpid(self, span, event):
        should_add = True
        for tid_pid_proc in span['tidPid']:
            if tid_pid_proc['tid'] == event['vtid'] and tid_pid_proc['pid'] == event['vpid'] and tid_pid_proc['procname'] == event['procname']:
                should_add = False
                break
        return should_add

    # This function process an event and change the spans directory
    def process_event(self, event, index):
        event_tag = event['msgTag']
        root_tag = self.find_tag(event['msgTag'], 0)+'/'
        if event_tag.count('"tag":') == 1:  # This is a root span
            if root_tag in self.spans.keys():
                self.spans[root_tag]['count'] += 1
                self.spans[root_tag]['duration'] = event['timestamp'] - \
                    self.spans[root_tag]['timestamp']
                #self.spans[root_tag]['reqrepseq'].append({'neme': '{}/{}'.format(event['msgType'], event['procname']), 'timestamp':event['timestamp']})
                if self.should_add_tidpid(self.spans[root_tag], event):
                    self.spans[root_tag]['tidPid'].append(
                        {'tid': event['vtid'], 'pid': event['vpid'], 'procname': event['procname']})
            else:
                self.spans[root_tag] = {
                    'spanId': root_tag,
                    'count': 1, 'timestamp': event['timestamp'],
                    'duration': 0,
                    'tidPid': [{'tid': event['vtid'], 'pid':event['vpid'], 'procname': event['procname']}],
                    # 'reqrepseq': [{'neme': '{}/{}'.format(event['msgType'], event['procname']), 'timestamp':event['timestamp']}],
                    # 'alias': 1,
                    # 'max_alias':1
                }

        elif event_tag.count('"tag":') > 1:  # This is a subspan
            if root_tag in self.spans.keys():
                should_consider, subspan_tag = self.should_consider_subspan(
                    event['msgTag'], root_tag)
                if should_consider and subspan_tag not in self.spans[root_tag].keys():
                    # self.spans[root_tag]['alias']+=1
                    # alias = self.spans[root_tag]['alias']
                    self.spans[root_tag][subspan_tag] = {
                        'spanId': subspan_tag,
                        'count': 1, 'timestamp': event['timestamp'], 'duration': 0,
                        'tidPid': [{'tid': event['vtid'], 'pid':event['vpid'], 'procname': event['procname']}],
                        # 'alias':alias
                        }
                    #self.spans[root_tag]['reqrepseq'].append({'neme': '{}/{}'.format(event['msgType'], event['procname']), 'timestamp':event['timestamp']})
                elif should_consider and subspan_tag in self.spans[root_tag].keys():
                    self.spans[root_tag][subspan_tag]['count'] += 1
                    self.spans[root_tag][subspan_tag]['duration'] = event['timestamp'] - \
                        self.spans[root_tag][subspan_tag]['timestamp']
                    if self.should_add_tidpid(self.spans[root_tag][subspan_tag], event):
                        self.spans[root_tag][subspan_tag]['tidPid'].append(
                            {'tid': event['vtid'], 'pid': event['vpid'], 'procname': event['procname']})
                    #self.spans[root_tag]['reqrepseq'].append({'neme': '{}/{}'.format(event['msgType'], event['procname']), 'timestamp':event['timestamp']})
            else:
                self.trace_collection[index]['status']= False


    # This function reads all events in the trace and makes the spans directory
    def make_spans(self):
        for index, event in enumerate(self.trace_collection):
            self.trace_collection[index]['status']= True
            if event['name'] == 'msgTrace:reqrep':
                self.process_event(event, index)

    # This function remove incomplete spans and subspans from spans directory.
    # The start or end of an span may not locate in the trace
    def remove_incomplete_spans(self):
        span_ids = [span_id for span_id in self.spans.keys()]
        for span_id in span_ids:
            if self.spans[span_id]['count'] < 4:
                del self.spans[span_id]
            else:
                sub_span_ids = [
                    key for key in self.spans[span_id] if key.startswith('"')]
                for sub_span_id in sub_span_ids:
                    if sub_span_id in self.spans[span_id].keys() and self.spans[span_id][sub_span_id]['count'] < 4:
                        del self.spans[span_id][sub_span_id]
                        childs_of_sub_span = [
                            key for key in self.spans[span_id] if key.startswith(sub_span_id)]
                        for child in childs_of_sub_span:
                            del self.spans[span_id][child]

    # This function returns spans directory

    def get_spans(self):
        return self.spans, self.trace_collection
