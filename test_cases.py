import nltk
import re
import math
import numpy
from nltk.tokenize.texttiling import TextTilingTokenizer
from parameter import Parameter
from texttiling import *
from evaluation import *
from output import *

def baseline_test():
    def read_target(fname):
        targets = []
        text = ""
        with open(fname) as f:
            content = f.readlines()
            for line in content:
                text+=line
                if line[0:2] == '[[':
                    targets.append((line.split('[[')[1]).split(']')[0].lower())
        f.close()
        lowercase_text = text.lower()
        tt = nltk.tokenize.texttiling.TextTilingTokenizer(w=40, k=28, demo_mode=True)
        paragraph_breaks = tt._mark_paragraph_breaks(text)
        text_length = len(lowercase_text)

        # Remove punctuation
        nopunct_text = ''.join(c for c in lowercase_text
                               if re.match("[a-z\-\' \n\t]", c))

        nopunct_par_breaks = tt._mark_paragraph_breaks(nopunct_text)
        tokseqs = tt._divide_to_tokensequences(nopunct_text)

        target_boundry = find_target_boundry(tt, nopunct_text, targets)
        return target_boundry

    lecs = [
            "data/Microeconomics/Microecon_1.train", "data/Microeconomics/Microecon_2.train", 
            "data/Microeconomics/Microecon_3.train", "data/Microeconomics/Microecon_4.train", 
            "data/Microeconomics/Microecon_5.train", 
            "data/Microeconomics/Microecon_18.train", "data/Microeconomics/Microecon_19.train", 
            "data/Microeconomics/Microecon_20.train", "data/Microeconomics/Microecon_21.train", 
            "data/Microeconomics/Microecon_22.train", "data/Microeconomics/Microecon_23.train", 
            "data/Microeconomics/Microecon_24.train", "data/Microeconomics/Microecon_25.train", 
            "data/Microeconomics/Microecon_26.train"]
    lecs2 = [
            "data/Psychology/Psych_1.train", "data/Psychology/Psych_2.train", 
            "data/Psychology/Psych_3.train", "data/Psychology/Psych_4.train", 
            "data/Psychology/Psych_5.train", "data/Psychology/Psych_6.train", 
            "data/Psychology/Psych_7.train", "data/Psychology/Psych_8.train", 
            "data/Psychology/Psych_9.train", "data/Psychology/Psych_10.train", 
            "data/Psychology/Psych_11.train", "data/Psychology/Psych_12.train", 
            "data/Psychology/Psych_13.train", "data/Psychology/Psych_14.train", 
            "data/Psychology/Psych_15.train", "data/Psychology/Psych_16.train", 
            "data/Psychology/Psych_17.train",  
            "data/Psychology/Psych_18.train", "data/Psychology/Psych_19.train", 
            "data/Psychology/Psych_20.train", "data/Psychology/Psych_21.train", 
            "data/Psychology/Psych_22.train", "data/Psychology/Psych_23.train", 
            "data/Psychology/Psych_24.train"]
    lecs3 = [
            "data/Engineering_Dynamics/EngDyn_1.train", "data/Engineering_Dynamics/EngDyn_2.train", 
            "data/Engineering_Dynamics/EngDyn_3", "data/Engineering_Dynamics/EngDyn_4", 
            "data/Engineering_Dynamics/EngDyn_5", "data/Engineering_Dynamics/EngDyn_6", 
            "data/Engineering_Dynamics/EngDyn_7", "data/Engineering_Dynamics/EngDyn_8", 
            "data/Engineering_Dynamics/EngDyn_9", "data/Engineering_Dynamics/EngDyn_10", 
            "data/Engineering_Dynamics/EngDyn_11", "data/Engineering_Dynamics/EngDyn_12", 
            "data/Engineering_Dynamics/EngDyn_13", "data/Engineering_Dynamics/EngDyn_14", 
            "data/Engineering_Dynamics/EngDyn_15", "data/Engineering_Dynamics/EngDyn_16", 
            "data/Engineering_Dynamics/EngDyn_17",  
            "data/Engineering_Dynamics/EngDyn_18"]
    p=[]
    r=[]
    f1=[]
    for lec in lecs2:
        target_boundry = read_target(lec)
        print (lec, len(target_boundry))
        a,b,c = baseline(target_boundry)
        p.append(a)
        r.append(b)
        f1.append(c)
    print (sum(p)/len(p), sum(r)/len(r), sum(f1)/len(f1))


def test_best_setup(text, targets):
    tt = nltk.tokenize.texttiling.TextTilingTokenizer(w=40, k=28, demo_mode=True)

    new_text, s, ss, d, b,t = tokenize(tt, text, targets, 80, 9, True, 0, 0, 0)
    plot(s, ss, d, b, t)

    precision, recall, f1 = evaluate(b,t)
    print(precision, recall, f1)
    baseline(t)
    return new_text

def test_cue_word(text, targets):
    for i in range(50, 100, 10):
        for j in range(7,12):
            for k in [True, False]:
                tt = nltk.tokenize.texttiling.TextTilingTokenizer(w=38, k=23, demo_mode=True)
                new_text, s, ss, d, b,t = tokenize(tt, text, targets, i, j, k)
                # plot(s, ss, d, b, t)
                precision, recall, f1 = evaluate(b,t)
                print('percent: ', i, 'distance: ', j, 'cue', k, precision, recall, f1)

def test_w_k(text, targets):
    # test for w and k
    for ww in range(30, 50, 2):
        for kk in range(20,30):
            tt = nltk.tokenize.texttiling.TextTilingTokenizer(w = ww, k=kk, demo_mode=True)
            new_text, s, ss, d, b,t = tokenize(tt, text, targets, 70, 9)
            # plot(s, ss, d, b, t)
            precision, recall, f1 = evaluate(b,t)
            print('w: ', ww, 'k: ', kk, precision, recall, f1)


def test_np(text, targets):
    for i in range(70, 90, 10):
        for j in range(7,11):
            for k in [0,0.05,0.1,0.15,0.2,0.25,0.3]:
                tt = nltk.tokenize.texttiling.TextTilingTokenizer(w=38, k=23, demo_mode=True)
                new_text, s, ss, d, b,t = tokenize(tt, text, targets, i, j, False, k)
                # plot(s, ss, d, b, t)
                precision, recall, f1 = evaluate(b,t)
                print('percent: ', i, 'distance: ', j, 'np_percent', k, precision, recall, f1)

def test_all(text, targets):
    parameters = []
    # for w in range(36, 48, 2):
    #     for k in range(20,30, 4):
    #         for percentile in range(70, 90, 10):
    #             for boundary_diff in range(7,10):
    #                 for np_percent in [0]:
    #                     for cue_percent in [0]:
    for w in [40]:
        for k in [28]:
            for percentile in [80]:
                for boundary_diff in [9]:
                    for np_percent in [ 0.8, 0.9, 1.0]:
                        for cue_percent in [0]:
                    # for np_percent in [0,0.05]:
                    #     for cue_percent in [0,0.05]:
                            tt = nltk.tokenize.texttiling.TextTilingTokenizer(w=w, k=k, demo_mode=True)
                            new_text, s, ss, d, b,t = tokenize(tt, text, targets, percentile, boundary_diff, False, np_percent, cue_percent)
                            precision, recall, f1 = evaluate(b,t)
                            parameters.append(Parameter(k, w, percentile, boundary_diff, np_percent, cue_percent, precision, recall, f1))
                            print (w, k, percentile, boundary_diff, np_percent, cue_percent, precision, recall, f1)
    sorted_para = sorted(parameters, key=lambda parameter: parameter.f1, reverse = True)
    print (sorted_para)
    f = open('test_all.out','w') 
    for para in sorted_para:
        output = str(para) + '\n'
        f.write(output)
    f.close()
    return new_text

def test_cuewords_weight(lecs):
    b0 = []
    t0 = []
    b10 = []
    t10 = []
    b20 = []
    t20 = []
    b30 = []
    t30 = []
    b40 = []
    t40 = []
    bf = []
    tf = []
    for lec in lecs:
        text = ""
        targets = []
        with open(lec) as f:
            content = f.readlines()
            for line in content:
                text+=line
                if line[0:2] == '[[':
                    # print (line)
                    targets.append((line.split('[[')[1]).split(']')[0].lower())
            f.close()
            tt = nltk.tokenize.texttiling.TextTilingTokenizer(w=38, k=23, demo_mode=True)
            new_text, s, ss, d, b,t = tokenize(tt, text, targets, 75, 8, False, 0, 0)
            b0.extend(b)
            t0.extend(t)
            #print t
            new_text, s, ss, d, b,t = tokenize(tt, text, targets, 75, 8, False, 0, 0.1)
            b10.extend(b)
            t10.extend(t)
            #print t
            new_text, s, ss, d, b,t = tokenize(tt, text, targets, 75, 8, False, 0, 0.2)
            b20.extend(b)
            t20.extend(t)
            new_text, s, ss, d, b,t = tokenize(tt, text, targets, 75, 8, False, 0, 0.3)
            b30.extend(b)
            t30.extend(t)
            new_text, s, ss, d, b,t = tokenize(tt, text, targets, 75, 8, False, 0, 0.4)
            b40.extend(b)
            t40.extend(t)
            new_text, s, ss, d, b,t = tokenize(tt, text, targets, 75, 8, True, 0, 0)
            bf.extend(b)
            tf.extend(t)
            #print t
    print 0, evaluate(b0, t0)
    print 10, evaluate(b10, t10)
    print 20, evaluate(b20, t20)
    print 30, evaluate(b30, t30)
    print 40, evaluate(b40, t40)
    print "filter", evaluate(bf, tf)

# baseline_test()