import numpy as np
import sys
import re
import math
import string
from sklearn import tree
from sklearn.feature_selection import chi2
from nltk.metrics.segmentation import windowdiff
from nltk.metrics.segmentation import pk

training_filenames = ["l1", "l2", "l3", "l4", "l5", "l6", "l7", "l8", "l9", "l10", "l11", "l12", "l13", "l14", "l15", "l16", "l17"]

Sentences = []
Tags = []
Word_dict = {}
Word_count = {}
total_words = 0

Word_list = []

for filename in training_filenames:
	with open(filename) as f:
		prev = 2
		sentence = ""
		data = f.readlines()
		for line in data:
			#print line
			for i in range(len(line)):
				if line[i] == '?' or line[i] == '.':
					sentence = sentence.lower()
					Sentences.append(sentence.split(' '))
					for w in Sentences[len(Sentences) - 1]:
						if w not in Word_count:
							Word_count[w] = 0
							Word_list.append(w)
						Word_count[w] += 1
						total_words += 1
					sentence = ""
					if prev == 2:
						Tags.append(1)
						prev = 1
					elif i+1 != len(line) and line[i+1] == '/':
						Tags.append(0)
						prev = 2
					else:
						Tags.append(0)
						prev = 0
				elif line[i] == "\n" and sentence != "":
					sentence += " "
				elif line[i] not in string.punctuation and line[i]!='\n':
					if sentence != "" or line[i] != " ":
						sentence += line[i]
#print Word_dict
'''
threshold = 0.5 * total_words/len(Word_count)
for a in Word_list:
	if Word_count[a] < threshold:
		Word_list.remove(a)
'''
for i in range(len(Word_list)):
	Word_dict[Word_list[i]] = i
#print len(Word_list), len(Word_dict)
feature_vector = np.zeros(shape = (len(Sentences), 1 * len(Word_dict)), dtype = np.float32)
prev = [0 for i in range(len(Word_dict))]
prev[0] = 1
for i in range(len(Tags)):
	temp = [0 for k in range(len(Word_dict))]
	for w in Sentences[i]:
		if w in Word_dict:
			temp[Word_dict[w]] = 1
	for j in range(len(Word_dict)):
		feature_vector[i][j] = temp[j] or prev[j]

	prev = temp
#clf = tree.DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
#clf = ExtraTreesClassifier(criterion='entropy', random_state = 0)
#clf.fit(feature_vector, Tags)
#importance = clf.feature_importances_
importance, pval = chi2(feature_vector, Tags)

#print sorted(importance)
sorted_features = sorted(range(len(importance)), key=lambda i:importance[i])
#print sorted_features
f1 = open('chi2_cuewords','w') 
for i in range(len(importance)-1, len(importance)-31, -1):
	#f2.write(Word_list[sorted_features[i]] +' '+ str(importance[sorted_features[i]]) + ' ' + str(pval[sorted_features[i]]) + '\n')
	f1.write(Word_list[sorted_features[i]] + '\n')
