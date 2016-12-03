#!/usr/bin/python
import sys
import json
import io
import time
import numpy as np
import StringIO
import nltk
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support
from nltk.metrics import *
from vaderSentiment.vaderSentiment import sentiment as vaderSentiment
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
from sklearn import preprocessing, neighbors

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    theStems = []
    for item in tokens:
        theStems.append(stemmer.stem(item))
    return theStems

def getKNNFeatures(reviewText,i1):
    reviewText = reviewText.strip()
    soln = reviewText.split()
    eachFeatureSet = []
    vs = vaderSentiment(reviewText.encode("utf-8"))
    eachFeatureSet.append(vs['compound'])
    eachFeatureSet.append(vs['neu'])
    eachFeatureSet.append(vs['pos'])
    eachFeatureSet.append(vs['neg'])
    token = word_tokenize(reviewText)
    # c = Counter(token)
    posi = 0
    nega = 0

    for word in token:
        if word in positive:
            posi += 1
        if word in negative:
            nega += 1

    eachFeatureSet.append(posi)
    eachFeatureSet.append(nega)
    return eachFeatureSet
    # for disWord in distinct:
    #     if disWord in c:
    #         eachFeatureSet.append(c[disWord])
    #     else:
    #         eachFeatureSet.append(0)

    # eachFeatureSet.append(len(reviewText))

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
        if i >= 20000:
            break
        a = json.loads(line)
        for word in a['text'].split():
            distinct.add(word)

        FeatureSet.append(getKNNFeatures(a['text'],i))
        token_dict.append(a['text'].lower())
        knnY.append(a['stars'])
        i+=1
        # print i
f.close()

tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=True, encoding=u'utf-8',stop_words='english',ngram_range=(1, 2), max_df=1.0, min_df=1, max_features=100, binary=False, use_idf=True, smooth_idf=True, sublinear_tf=True)
knnX = tfidf.fit_transform(token_dict).toarray()

for x in range(len(knnX)):
    np.append(knnX[x],FeatureSet[x])
    # print knnX[x]

X = np.array(knnX[:15000])
y = np.array(knnY[:15000])
test_X = np.array(knnX[15000:])
test_y = np.array(knnY[15000:])
clf = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
clf.fit(X,y)

predict = clf.predict(test_X)
print(precision_recall_fscore_support(test_y,predict,average='weighted'))

