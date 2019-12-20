import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from math import e
import pickle
from sklearn import svm
import numpy as np
import timeit
import random




neg = open("./neg.txt", "r")
lines = [line.rstrip('\n') for line in neg]
neg_words = [[word for word in word_tokenize(line)] for line in lines]

pos = open("./pos.txt", "r")
lines = [line.rstrip('\n') for line in pos]
pos_words = [[word for word in word_tokenize(line)] for line in lines] 
#pos_words[n] == ["this", "for", "example"]

temp = neg_words
neg_words = []
for line in temp:
    for word in line:
        neg_words.append(word)

temp = pos_words
pos_words = []
for line in temp:
    for word in line:
        pos_words.append(word)
        
all_words = pos_words+neg_words
most_common_words = nltk.FreqDist(all_words)

most_common_words = [word for word in most_common_words if 900>most_common_words[word] > 2]

def create_feature(rev):
    feature = []
    for word in most_common_words:
        feature.append(word in word_tokenize(rev))
    return feature


#un feature consiste en una liste de True y False, para cada una de las palabras mas comunes
#segun esten en la rev

#neg = open(r"./neg.txt", "r")
#pos = open(r"./pos.txt", "r")

#pos_lines = [line.rstrip('\n') for line in pos]
#neg_lines = [line.rstrip('\n') for line in neg]

#featureset = []
#count = 0
#for rev in pos_lines[:1000]:
 #   featureset.append((create_feature(rev), "pos"))
 #   count += 1
 #   print(count, " out of 1000")
#count = 0
#for rev in neg_lines[:1000]:
#    featureset.append((create_feature(rev), "neg"))
#    count += 1
#    print(count , " out of 1000")

"""import pickle

PARA ENTRENAR AL CLASIFICADOR DE NUEVO CON OTRA DATA

pickle_out = open("featureset.pickle","wb")
pickle.dump(featureset, pickle_out)
pickle_out.close()

import random

random.shuffle(featureset)

test_set = featureset[1900:]
train_set = featureset[:1900]

X = [x[0] for x in train_set]
y = [x[1] for x in train_set]

X = np.array(X)
y = np.array(y)

clf = svm.LinearSVC()
clf.fit(X, y)

pickle_out = open("LinSVCclf.pickle", "wb")
pickle.dump(clf, pickle_out)
pickle_out.close()"""


pickle_in = open("LinSVCclf.pickle","rb")
clf = pickle.load(pickle_in)



def sentiment(sentence):
    X = np.array(create_feature(sentence))
    X = X.reshape(1, -1)
    return clf.predict(X)[0]

