#!/usr/bin/python
import sys
import json
import io
import time
import numpy as np
import StringIO
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support
from nltk.metrics import *
from vaderSentiment.vaderSentiment import sentiment as vaderSentiment
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def getSVMFeatures(reviewText,i1):
    reviewText = reviewText.strip()
    soln = reviewText.split()
    eachFeatureSet = []
    posi = 0
    nega = 0
    special = 0
    
    vs = vaderSentiment(reviewText.encode("utf-8"))
    eachFeatureSet.append(vs['compound'])
    eachFeatureSet.append(vs['neu'])
    eachFeatureSet.append(vs['pos'])
    eachFeatureSet.append(vs['neg'])
    token = word_tokenize(reviewText)
    token = stem_tokens(token, stemmer)

    for word in token:
        if not re.match(r'^\w+$', word):
            special += 1
        if word in positive:
            posi += 1
        if word in negative:
            nega += 1

    eachFeatureSet.append(special)
    eachFeatureSet.append(posi)
    eachFeatureSet.append(nega)
    return eachFeatureSet

print (time.clock())
i=0
knnX = []
knnX2 = []
knnY = []
token_dict = []
distinct = set([])
FeatureSet = []
positive = set([])
negative = set([])
stemmer = PorterStemmer()

with open('positive.txt') as f:
    data = f.readlines()
    positive = set(data)

with open('negative.txt') as f:
    data = f.readlines()
    negative = set(data)

with open('yelp_academic_dataset_review.json') as f:
    for line in f:
        if i >= 100000:
            break
        a = json.loads(line)
        for word in a['text'].split():
            distinct.add(word)

        FeatureSet.append(getSVMFeatures(a['text'],i))
        token_dict.append(a['text'])
        knnY.append(a['stars'])
        i+=1
        # print i
f.close()

tfidf = TfidfVectorizer(lowercase=True, encoding=u'utf-8',stop_words='english',ngram_range=(1, 2), max_features=700, max_df=1.0, min_df=1, binary=False, use_idf=True, smooth_idf=True, sublinear_tf=True)
knnX = tfidf.fit_transform(token_dict).toarray()

for x in range(len(knnX)):
    np.append(knnX[x],FeatureSet[x])

X = np.array(knnX[:75000])
y = np.array(knnY[:75000])
test_X = np.array(knnX[75000:])
test_y = np.array(knnY[75000:])
clf = svm.SVC(C=1.0, cache_size=200, class_weight='balanced',coef0=0.0, degree=3, gamma='auto', kernel='linear',max_iter=-1, probability=False, random_state=None, shrinking=True,tol=0.001, verbose=False)
clf.fit(X,y)

predict = clf.predict(test_X)
print(precision_recall_fscore_support(test_y,predict,average='weighted'))

