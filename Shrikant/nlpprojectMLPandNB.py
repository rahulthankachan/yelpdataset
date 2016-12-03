import json
import pickle
import os
import time
import nltk
import numpy
from nltk import re, wordpunct_tokenize, defaultdict, NaiveBayesClassifier
from nltk.corpus import stopwords
from sklearn.metrics import f1_score
start_time = time.clock()
from nltk import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#from vaderSentiment.vaderSentiment import sentiment as vaderSentiment
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from random import shuffle

def get_review_length(review_text):
    list = review_text.split()
    return (len(list))


data = []
reviews = []
labels = []
features = []
corpus = []
labels = []
def readinput(n):
    total=0
    five = 0
    four = 0
    three = 0
    two = 0
    one = 0
    fives=0
    fours=0
    threes=0
    twos=0
    ones=0
    limit = int(n/5)
    print(limit)
    with open('C:\yelp_academic_dataset_review.json') as f:
        for line in f:
            a = json.loads(line)
            if total>=n:
                break
            if a['stars'] == 5:
                five +=1
                #print(five)
                if five <=limit:
                    fives +=1
                    reviews.append((a['text'],a['stars']))
                else:
                    continue

            if a['stars'] == 4:
                four += 1
                if four <=limit:
                    fours += 1
                    reviews.append((a['text'], a['stars']))
            if a['stars'] ==3:
                three +=1
                if three <=limit:
                    threes +=1
                    reviews.append((a['text'], a['stars']))
            if a['stars'] == 2:
                two += 1
                if two <=limit:
                    twos +=1
                    reviews.append((a['text'], a['stars']))
            if a['stars'] == 1:
                one += 1
                if one <= n / 5:
                    ones += 1
                    reviews.append((a['text'], a['stars']))

            #print(bigramReturner(a['text']))
            #reviews.append((a['text'],a['stars']))
            #print(total)
            total = fives+threes+twos+fours+ones
    shuffle(reviews)



def generateXandY():
    for i,j in reviews:
        corpus.append(i)
        labels.append(j)

def getfeatures():
    vectorizer = TfidfVectorizer(min_df=0.001,smooth_idf=True ,stop_words='english',lowercase=True,max_features=200, ngram_range=(1,1))
    #vectorizer = CountVectorizer(min_df=0.001,stop_words='english',lowercase=True,max_features=100, ngram_range=(1,2))
    X = vectorizer.fit_transform(corpus).toarray()
    print(vectorizer)
    vs = SentimentIntensityAnalyzer()
    for i in range(len(corpus)):
        ss= vs.polarity_scores(corpus[i])
        #for k in sorted(ss):
        #    numpy.append(X[i],ss[k])
        #numpy.append(X[i],get_review_length(corpus[i]))
        numpy.append(X[i],sorted(ss))
        numpy.append(X[i],get_review_length(corpus[i]))
    return X

def NaiveBayes():

    clf = GaussianNB()
    clf.fit(train_data,train_labels)
    y_pred = clf.predict(test)
    print("Gaussian Naive Bayes F1-score: ", f1_score(test_labels, y_pred, average='weighted'))
    print(time.clock()-start_time , "sec")
    print(classification_report(test_labels,y_pred))


    score = cross_val_score(clf,train_data,train_labels,cv=4,scoring="f1_macro")

    print(score)

    clf = BernoulliNB()
    clf.fit(train_data,train_labels)
    print(clf)
    y_pred = clf.predict(test)
    print("Bernoulli Naive Bayes F1-score: ", f1_score(test_labels, y_pred, average='weighted'))
    print(classification_report(test_labels,y_pred))

    clf1 = MultinomialNB()
    clf1.fit(train_data,train_labels)
    y_pred =clf1.predict(test)
    print("Multinomial NB",f1_score(test_labels, y_pred, average='weighted'))
    print(classification_report(test_labels,y_pred))


def MLP(solver,aplha):
    print("MLP lbfgs alpha", aplha)
    clf2 = MLPClassifier(solver=solver, alpha=aplha)
    clf2.fit(train_data,train_labels)
    y_pred=clf2.predict(test)
    print("F1-score", f1_score(test_labels, y_pred, average='weighted'))
    print(classification_report(test_labels,y_pred))

readinput(10000)
generateXandY()
X=getfeatures()
train_data = X[:7000]
train_labels = labels[:7000]
test = X[7000:]
test_labels = labels[7000:]
NaiveBayes()
MLP("sgd",1e-2)


print(time.clock()-start_time , "sec")
