import nltk
import re
import math
import numpy
import copy
import json
from nltk.tokenize.texttiling import TextTilingTokenizer
from nltk.corpus import wordnet

def _wn_block_comparison(tokseqs, token_table):
    "Implements the block comparison method"
    TT_K = 30

    def blk_frq(tok, block):
        # get all synonyms for verb form of tok
        synonyms = [tok]
        for synset in wordnet.synsets(tok, pos='v'):
            for hypernym in synset.hypernyms():
                synonyms += hypernym.lemma_names()
            synonyms += synset.lemma_names()
        synonyms = list(set(synonyms)) # remove duplicates

        # sum all occurences of word and its synonyms into one sum
        ts_occs = []
        for word in synonyms:
            if word not in token_table:
                continue
            ts_occs += filter(lambda o: o[0] in block,
                             token_table[word].ts_occurences)
        freq = sum([ts_occ[1] for ts_occ in ts_occs])
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

        for t in token_table:
            score_dividend += blk_frq(t, b1)*blk_frq(t, b2)
            score_divisor_b1 += blk_frq(t, b1)**2
            score_divisor_b2 += blk_frq(t, b2)**2
        try:
            score = score_dividend/math.sqrt(score_divisor_b1*
                                             score_divisor_b2)
        except ZeroDivisionError:
            pass # score += 0.0

        gap_scores.append(score)

    return gap_scores