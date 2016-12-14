import nltk
import re
import math
import numpy
import copy
import json
from nltk.tokenize.texttiling import TextTilingTokenizer
from textblob import TextBlob

def _create_ngrams_table(tokseqs, n = 2):
    ngram_table = {}
    current_tok_seq = 0
    tt = nltk.tokenize.texttiling.TokenTableField
    for ts in tokseqs:
        ts_tb = TextBlob(' '.join([word[0] for word in ts.wrdindex_list]))
        ngrams = nltk.bigrams([word[0] for word in ts.wrdindex_list])
        if n == 3:
            ngrams = nltk.trigrams([word[0] for word in ts.wrdindex_list])
        for ngram in ngrams:
            if ngram in ngram_table:
                ngram_table[ngram].total_count += 1

                if ngram_table[ngram].last_tok_seq != current_tok_seq:
                    ngram_table[ngram].last_tok_seq = current_tok_seq
                    ngram_table[ngram].ts_occurences.append([current_tok_seq,1])
                else:
                    ngram_table[ngram].ts_occurences[-1][1] += 1
            else: #new word
                ngram_table[ngram] = tt(first_pos=0,
                                        ts_occurences=[[current_tok_seq,1]],
                                        total_count=1,
                                        par_count=1,
                                        last_par=0,
                                        last_tok_seq=current_tok_seq)
        current_tok_seq += 1

    return ngram_table