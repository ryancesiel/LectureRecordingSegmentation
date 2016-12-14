import nltk
import copy
import json
import re
import math
import numpy
from matplotlib import pylab
from parameter import Parameter


def plot(s, ss, d, b, t):
    # print (b, t)
    pylab.xlabel("Sentence Gap index")
    pylab.ylabel("Gap Scores")
    pylab.plot(range(len(s)), s, label="Gap Scores")
    pylab.plot(range(len(ss)), ss, label="Smoothed Gap scores")
    pylab.plot(range(len(d)), d, label="Depth scores")
    pylab.stem(range(len(b)), b)
    pylab.stem(range(len(t)), t,  '-.')
    pylab.legend()
    pylab.show()

def export(new_text):
    f = open('test.out','w') 
    index = 0
    for para in new_text:
        index+=1
        output = para + '\n\n\n' + '<<BREAK>>' + '\n\n\n'
        f.write(output)
    f.close()

def getPredictedTimestamps(timestamps_file, new_text):
    # (sentence beginning, start time)
    sentence_starts = []
    with open(timestamps_file) as f:
        iter_f = iter(f)
        prev_line = ''
        for line in iter_f:
            if str(line[0]).isupper() and prev_line[0] == '0':
                sentence_starts.append((line[:-1], prev_line[:8]))
            prev_line = line[:8]

    # { 
    #   'Heading1': {
    #       'text': 'text within boundary',
    #       'start': '##:##:##',
    # }
    segment_timestamps = {}
    n = 1
    for segment in new_text:
        segment_begin = segment.split('.')[0]
        # print '------------------------------------'
        # print 'Looking for segment:', segment_begin
        for candidate in sentence_starts:
            if candidate[0][:-1] in segment_begin:
                # print 'Found matching candidate:', candidate[1], '\n', candidate[0]
                segment_timestamps['Heading'+str(n)] = {}
                segment_timestamps['Heading'+str(n)]['text'] = segment
                segment_timestamps['Heading'+str(n)]['start'] = candidate[1]
                n += 1
                break
    return segment_timestamps

def getGoldTimestamps(gold_times_file):
    # { 
    #   'Heading1': {
    #       'heading': 'actual heading title',
    #       'start': '##:##:##',
    #       'end': '##:##:##' }
    # }
    gold_timestamps = {}
    with open(gold_times_file) as f:
        iter_f = iter(f)
        n = 1
        for line in iter_f:
            if line[:9] == 'Heading: ':
                gold_timestamps['Heading'+str(n)] = {}
                head = line[9:-1]
                gold_timestamps['Heading'+str(n)]['heading'] = head
                line = next(iter_f)
                start = line[7:-1]
                gold_timestamps['Heading'+str(n)]['start'] = start
                line = next(iter_f)
                end = line[5:-1]
                gold_timestamps['Heading'+str(n)]['end'] = end
                n += 1
    return gold_timestamps

def generateJSONTimestamps(gold_times_file, timestamps_file, new_text):

    gold_timestamps = getGoldTimestamps(gold_times_file)
    predicted_timestamps = getPredictedTimestamps(timestamps_file, new_text)

    with open('gold_JSON.json', 'w') as out_f:
        json.dump(gold_timestamps, out_f)
    out_f.close()

    with open('predicted_JSON.json', 'w') as out_f:
        json.dump(predicted_timestamps, out_f)
    out_f.close()
