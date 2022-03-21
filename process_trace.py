##########################################################################
## Read trace files in trace compass and make a database of events      ##
## Iman Kohyarnejadfard                                                 ##
## Polytechnique Montréal                                               ##
## process_trace.py                                                     ##
##########################################################################

import time
import json
import gc
from java.util.function import Function
import csv
from LoadReqRepSpans import LoadReqRepSpans

loadModule('/TraceCompass/Analysis');
loadModule('/TraceCompass/DataProvider');
loadModule('/TraceCompass/Trace');
loadModule('/TraceCompass/Utils');
loadModule('/TraceCompass/View');

csv_path='/home/iman/Desktop/ust_events.csv'

start = time.time()
# Create an analysis named userv_msg_seq.py
analysis = createScriptedAnalysis(getActiveTrace(), "ust_events_j.py")

def strToVarargs(str):
	return [str]

if (analysis == None):
	print("Trace is null")
	exit()

# Get the analysis's state system so we can fill it, false indicates to create a new state system even if one already exists, true would re-use an existing state system
ss = analysis.getStateSystem(False)
 
csv.register_dialect('myDialect', delimiter='/', quoting=csv.QUOTE_NONE)  
# The analysis itself is in this function
def runAnalysis():
    # Get the event iterator for the trace
    iter = analysis.getEventIterator()
	
    # Parse all events
    print("Step 1: process UST events...")
    f = open(csv_path, 'w')
    with f:
    	myFields = ['name', 'timestamp','msgTag','vtid','vpid','procname']
    	writer = csv.DictWriter(f, fieldnames=myFields)
    	writer.writeheader()
    	while (iter.hasNext()):
			event = iter.next()
			ust_event={}
			
			ust_event['name'] = event.getName()
			ust_event['timestamp'] = event.getTimestamp().toNanos()
			ust_event['msgTag'] = getEventFieldValue(event, "msgTag")
			ust_event['vtid'] = getEventFieldValue(event, "tid")
			ust_event['vpid'] = getEventFieldValue(event, "pid")
			ust_event['procname'] = getEventFieldValue(event, "context._procname")
			
			writer.writerow(ust_event)
	print('Step 1: UST events are processed.')
            
        
			    
def make_spans(csv_path):
    rows = []
    print('Step 2: Load UST data...')
    with open(csv_path) as myFile:
        reader = csv.DictReader(myFile)
        for row in reader:
            ust_event = {}
            ust_event['name'] = row['name']
            ust_event['timestamp'] = float(row['timestamp'])
            ust_event['msgTag'] = row['msgTag']
            ust_event['vtid'] = int(row['vtid'])
            ust_event['vpid'] = int(row['vpid'])
            ust_event['procname'] = row['procname']

            rows.append(ust_event)
    print('Step 2: UST data is loaded.')

    # Make spans
    print('Step 3: Extracting the spans is started...')
    lrr = LoadReqRepSpans(rows)
    lrr.make_spans()
    lrr.remove_incomplete_spans()
    spans = lrr.get_spans()
    # Create spans.json and spans_events.json
    with open('spans.json', 'w') as json_file:
        json.dump(spans, json_file)
    print('Step 3: Spans are successfully extracted.')
    
runAnalysis()
spans = make_spans('events.csv')










