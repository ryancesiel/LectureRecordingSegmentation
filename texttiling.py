import nltk
import re
import math
import numpy
import copy
import json
from nltk.tokenize.texttiling import TextTilingTokenizer
from textblob import TextBlob
from ngrams import _create_ngrams_table
from noun_phrase import _np_block_comparison
from word_net import _wn_block_comparison
from cue_word import *


def tokenize(tt, text, targets, percent = 80, boundary_diff = 9, cue_filter = False, np_percent = 0, cue_percent = 0, verb_percent = 0, n_gram = 1):

    lowercase_text = text.lower()
    paragraph_breaks = tt._mark_paragraph_breaks(text)
    text_length = len(lowercase_text)

    # Remove punctuation
    nopunct_text = ''.join(c for c in lowercase_text
                           if re.match("[a-z\-\' \n\t]", c))

    nopunct_par_breaks = tt._mark_paragraph_breaks(nopunct_text)
    tokseqs = tt._divide_to_tokensequences(nopunct_text)

    target_boundry = find_target_boundry(tt, nopunct_text, targets)
    _divide_to_tokensequences(tt, nopunct_text)

    # # test stemmer
    # stemmer = PorterStemmer()
    # for ts in tokseqs:
    #     for wi in ts.wrdindex_list:
    #         print (wi)
    #         wi = (stemmer.stem(wi[0]),wi[1])

    # print (len(tokseqs))

    # START UNIGRAM TOKEN_TABLE
    # Filter stopwords
    if n_gram == 1:
        for ts in tokseqs:
            ts.wrdindex_list = [wi for wi in ts.wrdindex_list
                                if wi[0] not in tt.stopwords]

    def is_verb(word):
        return word == 'VB' or word == 'VBZ' or word == 'VBP' \
               or word == 'VBD' or word == 'VBN' or word == 'VBG'

    verb_tokseqs = []
    for ts in tokseqs:
        verb_ts = copy.deepcopy(ts)
        words_with_pos_tags = TextBlob(' '.join([wi[0] for wi in verb_ts.wrdindex_list])).tags

        verb_ts.wrdindex_list = [wi for i, wi in enumerate(verb_ts.wrdindex_list)
                                 if is_verb(words_with_pos_tags[i][1])]
        verb_tokseqs.append(verb_ts)

    token_table = tt._create_token_table(tokseqs, nopunct_par_breaks)
    # print (token_table)

    if n_gram > 1:
        token_table = _create_ngrams_table(tokseqs, n_gram)
        gap_scores = tt._block_comparison(tokseqs, token_table)
        smooth_scores = tt._smooth_scores(gap_scores)
        depth_scores = tt._depth_scores(smooth_scores)
    else:
    # Lexical score determination
        gap_scores = tt._block_comparison(tokseqs, token_table)
        np_gap_scores = _np_block_comparison(tokseqs, [])
        verb_gap_scores = _wn_block_comparison(verb_tokseqs, token_table)

        smooth_scores = tt._smooth_scores(gap_scores)
        np_smooth_scores = tt._smooth_scores(np_gap_scores)
        verb_smooth_scores = tt._smooth_scores(verb_gap_scores)
        # End of Lexical score Determination

        # Boundary identification
        depth_scores = tt._depth_scores(smooth_scores)
        # print depth_scores
        np_depth_scores = tt._depth_scores(np_smooth_scores)
        verb_depth_scores = tt._depth_scores(verb_smooth_scores)

        for idx, np_ds in enumerate(np_depth_scores):
            depth_scores[idx] = depth_scores[idx] * (1-np_percent) + np_ds * np_percent

        for idx, verb_ds in enumerate(verb_depth_scores):
            depth_scores[idx] = depth_scores[idx] * (1-verb_percent) + verb_ds * verb_percent

    segment_boundaries = identify_boundaries(tt, depth_scores, tokseqs, percent, boundary_diff, cue_filter, cue_percent)

    normalized_boundaries = tt._normalize_boundaries(text,
                                                       segment_boundaries,
                                                       paragraph_breaks)

    # End of Boundary Identification
    segmented_text = []
    prevb = 0

    for b in normalized_boundaries:
        if b == 0:
            continue
        segmented_text.append(text[prevb:b])
        prevb = b

    if prevb < text_length: # append any text that may be remaining
        segmented_text.append(text[prevb:])

    if not segmented_text:
        segmented_text = [text]

    return segmented_text, gap_scores, smooth_scores, depth_scores, segment_boundaries, target_boundry

def find_target_boundry(tt, text, targets):

    w = tt.w
    wrdindex_list = []
    matches = re.finditer("\w+", text)
    # a = 0
    for match in matches:
        # a+=1
        wrdindex_list.append(match.group())
    target_boundry = [0] * int(len(wrdindex_list)/tt.w)
    # print (len(target_boundry))
    for t in targets:
        t =  ''.join(c for c in t
                           if re.match("[a-z\-\' \n\t]", c))
        words = t.split()
        lens = len(words)
        for i in range(len(wrdindex_list)):
            if wrdindex_list[i: i+lens] == words:
                target_boundry[int(i/tt.w)] = 1
                break
    return target_boundry

def _divide_to_tokensequences(tt, text):
    "Divides the text into pseudosentences of fixed size"
    w = tt.w
    wrdindex_list = []
    matches = re.finditer("\w+", text)
    a = 0
    for match in matches:
        a+=1
        wrdindex_list.append((match.group(), match.start()))
    # print (a, len(text))
    # print (len(range(0, len(wrdindex_list), w)))
    return 0

def identify_boundaries(tt, depth_scores, tokseqs, percent = 80, boundary_diff = 5, cue_filter = False, cue_percent = 0):
    """Identifies boundaries at the peaks of similarity score
    differences"""

    boundaries = [0 for x in depth_scores]

    avg = sum(depth_scores)/len(depth_scores)

    perc = numpy.percentile(numpy.array(depth_scores), percent)
    stdev = numpy.std(depth_scores)

    #SB: what is the purpose of this conditional?
    cutoff2 = avg-stdev/2.0
    cutoff = perc
    # print (perc, avg, stdev, cutoff)

    depth_tuples = sorted(zip(depth_scores, range(len(depth_scores))))

    depth_tuples.reverse()


    hp = list(filter(lambda x:x[0]>cutoff, depth_tuples))
    if cue_filter:
        hp = filter_cue_words(hp, tokseqs)

    hp = update_hp_by_cue_words(hp, tokseqs, cue_percent)
    hp = list(filter(lambda x:x[0]>cutoff, hp))
    for dt in hp:
        boundaries[dt[1]] = 1
        for dt2 in hp: #undo if there is a boundary close already
            if dt[1] != dt2[1] and abs(dt2[1]-dt[1]) < boundary_diff \
                   and boundaries[dt2[1]] == 1:
                boundaries[dt[1]] = 0
    return boundaries
