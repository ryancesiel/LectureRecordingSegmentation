from nltk.corpus import brown
from matplotlib import pylab
import nltk
from nltk.tokenize.texttiling import TextTilingTokenizer
import re
import math
import numpy
from summary import SimpleSummarizer
from nltk.stem.porter import *

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

def find_target_boundry(tt, text, targets):

    w = tt.w
    wrdindex_list = []
    matches = re.finditer("\w+", text)
    a = 0
    for match in matches:
        a+=1
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

def read_cue_words(fname):
    words = []
    with open(fname) as f:
        content = f.readlines()
        for line in content:
            words.append(line.split()[0])
    f.close()
    return set(words)

def filter_cue_words(hp, tokseqs):
    topic_begin = read_cue_words('topic_begin')
    topic_end = read_cue_words('topic_end')
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
    print (num, len(hp), len(new_hp))
    return new_hp

def identify_boundaries(tt, depth_scores, tokseqs, percent = 80, boundary_diff = 5, cue_filter = False):
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
    # print (hp)
    # print (new_hp)

    for dt in hp:
        boundaries[dt[1]] = 1
        for dt2 in hp: #undo if there is a boundary close already
            if dt[1] != dt2[1] and abs(dt2[1]-dt[1]) < boundary_diff \
                   and boundaries[dt2[1]] == 1:
                boundaries[dt[1]] = 0
    return boundaries

def tokenize(tt, text, targets, percent = 80, boundary_diff = 5, cue_filter = False):

    lowercase_text = text.lower()
    paragraph_breaks = tt._mark_paragraph_breaks(text)
    text_length = len(lowercase_text)

    # Tokenization step starts here

    # Remove punctuation
    nopunct_text = ''.join(c for c in lowercase_text
                           if re.match("[a-z\-\' \n\t]", c))
    # print (nopunct_text)
    nopunct_par_breaks = tt._mark_paragraph_breaks(nopunct_text)

    tokseqs = tt._divide_to_tokensequences(nopunct_text)
    
    target_boundry = find_target_boundry(tt, nopunct_text, targets)
    _divide_to_tokensequences(tt, nopunct_text)
    # print (nopunct_text)

    # # test stemmer
    # stemmer = PorterStemmer()
    # for ts in tokseqs:
    #     for wi in ts.wrdindex_list:
    #         print (wi)
    #         wi = (stemmer.stem(wi[0]),wi[1])

    # print (len(tokseqs))
    # Filter stopwords
    for ts in tokseqs:
        ts.wrdindex_list = [wi for wi in ts.wrdindex_list
                            if wi[0] not in tt.stopwords]

    token_table = tt._create_token_table(tokseqs, nopunct_par_breaks)
    # End of the Tokenization step

    # Lexical score determination
    gap_scores = tt._block_comparison(tokseqs, token_table)

    smooth_scores = tt._smooth_scores(gap_scores)
    # End of Lexical score Determination

    # Boundary identification
    depth_scores = tt._depth_scores(smooth_scores)

    # segment_boundaries = tt._identify_boundaries(depth_scores)

    segment_boundaries = identify_boundaries(tt, depth_scores, tokseqs, percent, boundary_diff, cue_filter)

    normalized_boundaries = tt._normalize_boundaries(text,
                                                       segment_boundaries,
                                                       paragraph_breaks)
    # print (depth_scores, segment_boundaries)
    # print (len(depth_scores), len(segment_boundaries))
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

def evaluate(boundary, target_boundry, err_tolerance = 2):
    tp = 0
    fp = 0
    fn = 0
    for i in range(1, len(target_boundry)):
        if target_boundry[i] != 0:
            flag = False
            for j in range(i-err_tolerance, i+err_tolerance):
                if boundary[j] != 0:
                    flag = True
                    break
            if flag:
                tp+=1
            else:
                fn+=1
    total = sum([0 if boundary[i]==0 else 1 for i in range(len(boundary))])
    total2 = sum([0 if target_boundry[i]==0 else 1 for i in range(len(target_boundry))])
    fp = total - tp
    # print (total, total2, tp, fp, fn)
    precision = 1.0 * tp / (fp + tp)
    recall = 1.0 * tp / (fn + tp)
    return precision, recall

def baseline(target_boundry):
    precisions = []
    recalls = []
    boundary_len = len(target_boundry)
    num_of_boundry = sum([0 if target_boundry[i]==0 else 1 for i in range(len(target_boundry))])

    for i in range(1000):
        baseline = numpy.random.choice(len(target_boundry), num_of_boundry)
        # print (baseline)
        baseline_boundry = [0] * boundary_len
        for b in baseline:
            baseline_boundry[b] = 1
        precision, recall = evaluate(baseline_boundry,target_boundry)
        precisions.append(precision)
        recalls.append(recall)
    # print (sum(precisions)/ len(precisions), sum(recalls)/ len(recalls))
    return baseline_boundry

def test_cue_word(text):
    # test for w and k
    for i in range(50, 100, 10):
        for j in range(7,12):
            for k in [True, False]:
                tt = nltk.tokenize.texttiling.TextTilingTokenizer(w=38, k=23, demo_mode=True)
                new_text, s, ss, d, b,t = tokenize(tt, text, targets, i, j, k)
                # plot(s, ss, d, b, t)
                precision, recall = evaluate(b,t)
                print('percent: ', i, 'distance: ', j, 'cue', k, precision, recall)

def test_w_k(text):
    # test for w and k
    for ww in range(30, 50, 2):
        for kk in range(20,30):
            tt = nltk.tokenize.texttiling.TextTilingTokenizer(w = ww, k=kk, demo_mode=True)
            new_text, s, ss, d, b,t = tokenize(tt, text, targets, 70, 9)
            # plot(s, ss, d, b, t)
            precision, recall = evaluate(b,t)
            print('w: ', ww, 'k: ', kk, precision, recall)

# filter_cue_words(0,0)
text = ""
targets = []
MIT_lec_1 = "MIT_lec_1.train"
MIT_lec_combined = "MIT_lec_combined.train"
test_name = "asr-output/eecs183-96.txt"
with open(MIT_lec_combined) as f:
    content = f.readlines()
    for line in content:
        text+=line
        if line[0:2] == '[[':
            # print (line)
            targets.append((line.split('[[')[1]).split(']')[0].lower())
f.close()
# print (targets)


test_cue_word(text)

# tt = nltk.tokenize.texttiling.TextTilingTokenizer(w=38, k=23, demo_mode=True)

# new_text, s, ss, d, b,t = tokenize(tt, text, targets, 70, 9)
# plot(s, ss, d, b, t)
# precision, recall = evaluate(b,t)
# print(precision, recall)
# baseline(t)



# # test for w and k
# for ww in range(30, 50, 2):
#     for kk in range(20,30):
#         tt = nltk.tokenize.texttiling.TextTilingTokenizer(w = ww, k=kk, demo_mode=True)
#         new_text, s, ss, d, b,t = tokenize(tt, text, targets, 70, 9)
#         # plot(s, ss, d, b, t)
#         precision, recall = evaluate(b,t)
#         print('w: ', ww, 'k: ', kk, precision, recall)

# ss = SimpleSummarizer()
# f = open('test.out','w') 
# index = 1
# for para in new_text:
#     index+=1
#     output = para + '\n\n\n' + '<<BREAK>>' + '\n\n\n'
#     print (index, ': ', ss.summarize(para, 1))
#     f.write(output)
# f.close()



