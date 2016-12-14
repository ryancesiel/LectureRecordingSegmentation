import nltk
import re
import math
import numpy
import copy
import json
from nltk.tokenize.texttiling import TextTilingTokenizer
from textblob import TextBlob

def _np_block_comparison(tokseqs, noun_phrases):
    "Implements the block comparison method"
    TT_K = 30

    # FROM _create_token_table
    np_table = {}
    current_tok_seq = 0
    for ts in tokseqs:
        ts_tb = TextBlob(' '.join([word[0] for word in ts.wrdindex_list]))
        for np in ts_tb.noun_phrases:
            if np in np_table:
                np_table[np]['total_count'] += 1

                if np_table[np]['last_tok_seq'] != current_tok_seq:
                    np_table[np]['last_tok_seq'] = current_tok_seq
                    np_table[np]['ts_occurences'].append([current_tok_seq,1])
                else:
                    np_table[np]['ts_occurences'][-1][1] += 1
            else: #new word
                np_table[np] = {
                    'ts_occurences': [[current_tok_seq,1]],
                    'total_count': 1,
                    'last_tok_seq': current_tok_seq
                }

        current_tok_seq += 1
    # END _create_token_table

    def blk_frq(np, block):
        ts_occs = filter(lambda o: o[0] in block,
                         np_table[np]['ts_occurences'])
        freq = sum([tsocc[1] for tsocc in ts_occs])
        return freq

    gap_scores = []
    numgaps = len(tokseqs)-1

    for curr_gap in range(numgaps):
        score_dividend, score_divisor_b1, score_divisor_b2 = 0.0, 0.0, 0.0
        score = 0.0
        #adjust window size for boundary conditions
        if curr_gap < TT_K-1:
            window_size = curr_gap + 1
        elif curr_gap > numgaps-TT_K:
            window_size = numgaps - curr_gap
        else:
            window_size = TT_K

        b1 = [ts.index
              for ts in tokseqs[curr_gap-window_size+1 : curr_gap+1]]
        b2 = [ts.index
              for ts in tokseqs[curr_gap+1 : curr_gap+window_size+1]]

        for np in np_table:
            score_dividend += blk_frq(np, b1)*blk_frq(np, b2)
            score_divisor_b1 += blk_frq(np, b1)**2
            score_divisor_b2 += blk_frq(np, b2)**2
        try:
            score = score_dividend/math.sqrt(score_divisor_b1*
                                             score_divisor_b2)
        except ZeroDivisionError:
            pass # score += 0.0

        gap_scores.append(score)

    return gap_scores