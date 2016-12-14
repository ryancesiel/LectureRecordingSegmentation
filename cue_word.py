import nltk
import re
import math
import numpy
import copy
import json
from nltk.tokenize.texttiling import TextTilingTokenizer
from textblob import TextBlob
from nltk.corpus import wordnet

def read_cue_words(fname):
    words = []
    with open(fname) as f:
        content = f.readlines()
        for line in content:
            words.append(line.split()[0])
    f.close()
    return set(words)

def filter_cue_words(hp, tokseqs):
    topic_begin = read_cue_words('results/chi2_cuewords')
    topic_end = read_cue_words('results/chi2_cuewords')
    # print (topic_begin, topic_end)
    num = 0
    new_hp = []
    for dt in hp:
        has_end_cue = False
        has_begin_cue = False
        # print (len(tokseqs[dt[1]].wrdindex_list))
        for wi in tokseqs[dt[1]].wrdindex_list:
            if wi[0] in topic_end:
                new_hp.append(dt)
                has_end_cue = True
                break
        if not has_end_cue:
            for wi in tokseqs[dt[1]+1].wrdindex_list:
                if wi[0] in topic_begin:
                    new_hp.append(dt)
                    has_begin_cue = True
                    break
        if has_begin_cue or has_end_cue:
            num += 1
    # print (num, len(hp), hp)
    return new_hp

def update_hp_by_cue_words(hp, tokseqs, cue_percent):
    topic_begin = read_cue_words('results/chi2_cuewords')
    topic_end = read_cue_words('results/chi2_cuewords')
    # print (topic_begin, topic_end)
    num = 0
    new_hp = []
    for dt in hp:
        has_end_cue = False
        has_begin_cue = False
        # print (len(tokseqs[dt[1]].wrdindex_list))
        for wi in tokseqs[dt[1]].wrdindex_list:
            if wi[0] in topic_end:
                new_hp.append(dt)
                has_end_cue = True
                break
        if not has_end_cue:
            for wi in tokseqs[dt[1]+1].wrdindex_list:
                if wi[0] in topic_begin:
                    new_hp.append(dt)
                    has_begin_cue = True
                    break
        if not has_begin_cue and not has_end_cue:
            new_hp.append((dt[0] * (1 - cue_percent), dt[1]))
    # print (len(new_hp), len(hp), hp, new_hp)
    return new_hp