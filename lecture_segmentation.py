import sys
import nltk
import re
import math
import numpy
import texttiling
from nltk.tokenize.texttiling import TextTilingTokenizer
from test_cases import *
from output import *

def main():
	text = ""
	targets = []

	# Intro Econ testfiles
	MIT_lec_1 = "data/MIT_lec_1.train"
	MIT_lec_3 = "data/MIT_lec_3.train"
	MIT_lec_combined = "data/MIT_lec_combined.train"
	MIT_all = "data/MIT_lec_all.train"
	# MIT_lec_testing = ["MIT_lec_18.train", "MIT_lec_19.train", "MIT_lec_20.train", "MIT_lec_21.train", "MIT_lec_22.train", "MIT_lec_23.train", "MIT_lec_24.train", "MIT_lec_25.train", "MIT_lec_26.train"]

	# Intro Psych testfiles
	Psych_lec_1 = "data/Lec2.train"
	gold_times_file = "data/Lec2_gold_times.txt"
	timestamps_file = "data/Lec2_timestamps.txt"

	# read lecture
	if len(sys.argv) < 2:
		fname = MIT_lec_1
	else:
		fname = sys.argv[1]
	with open(fname) as f:
	    content = f.readlines()
	    for line in content:
	        text+=line
	        if line[0:2] == '[[':
	            # print (line)
	            targets.append((line.split('[[')[1]).split(']')[0].lower())
	f.close()

	# run texttiling
	tt = nltk.tokenize.texttiling.TextTilingTokenizer(w=40, k=28, demo_mode=True)
	if len(sys.argv) < 3:
		new_text, s, ss, d, b,t = tokenize(tt, text, targets, 80, 9)
	elif sys.argv[2] == "Cue":
		if len(sys.argv) < 4:
			cue_percent = 0.5
		else:
			cue_percent = float(sys.argv[3])
		new_text, s, ss, d, b,t = tokenize(tt, text, targets, 80, 9, False, 0, cue_percent)
	elif sys.argv[2] == "Noun_Phrase":
		if len(sys.argv) < 4:
			np_percent = 0.5
		else:
			np_percent = float(sys.argv[3])
		# print np_percent
		new_text, s, ss, d, b,t = tokenize(tt, text, targets, 80, 9, False, np_percent)
	elif sys.argv[2] == "Verb":
		if len(sys.argv) < 4:
			verb_percent = 0.6
		else:
			verb_percent = float(sys.argv[3])
		# print verb_percent
		new_text, s, ss, d, b,t = tokenize(tt, text, targets, 80, 9, False, 0, 0, verb_percent)
	elif sys.argv[2] == "NGram":
		if len(sys.argv) < 4:
			n_gram = 2
		else:
			n_gram = int(sys.argv[3])
		# print n_gram
		new_text, s, ss, d, b,t = tokenize(tt, text, targets, 80, 9, False, 0, 0, 0, n_gram)
	else:
		print "wrong command"
		return
		
	# plot(s, ss, d, b, t)

	precision, recall, f1 = evaluate(b,t)
	print(precision, recall, f1)

	export(new_text)
	generateJSONTimestamps(gold_times_file, timestamps_file, new_text)



if __name__ == "__main__": main()
