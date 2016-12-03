import json
from nltk import tokenize
from nltk.corpus import stopwords
import nltk
import operator
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import precision_recall_fscore_support
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.classify import maxent
import time
from nltk import word_tokenize, pos_tag


#  Set of the StopWords identified  #####

stop = set(('and', 'or', 'not', 'the', ',', '(', ')', '!', 'are', 'I', 'i', 'am', 'to', 'at', 'a', 'will', 'was', 'by',
            'in', 'of'))

# Set of the StopWords identified  #####

# Feature selection ON/OFF (Configuration)
stemmer = PorterStemmer()

VADER_ON = True
POS_ON = True
STEMMING_ON = False
BAG_OF_WORDS_ON = True


complete = dict()

# Set of Positive words
POSITIVE_WORDS = set(open('positive.txt').read().split())
# Set of Negative Words
NEGATIVE_WORDS = set(open('negative.txt').read().split())


# Creating a bag of words
def get_word_count(tokens):
    mapping = dict()
    for t in tokens:
        if STEMMING_ON:
            t = stemmer.stem(t)
        if t in mapping:
            mapping[t] += 1
            complete[t] += 1
        else:
            if t not in stop:
                mapping[t] = 1
                complete[t] = 1
    return mapping


# gets the list which is send to the tfidf to learn
def get_tfidf_list(reviews):
    tfidf = list()
    for r in reviews:
        tfidf.append(r[1])
    return tfidf


# A tokenizer
def stem_tokens(tokens):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


# Tokenizes the text
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    if STEMMING_ON is True:
        tokens = stem_tokens(tokens)
    return tokens


# Get Token and Score
def get_token_score(features):
    mydict = dict()
    for i in range (0, len(features)):
        if features[i] > 0:
            mydict['TOKEN_'+str(i)] = True
    # print(mydict)
    return mydict


# Sentiment Scores using Vader
sid = SentimentIntensityAnalyzer()




# Reading all the reviews
i = 0

count = 40000  # Change this value to reduce the dataset
counts = [0, count, count, count, count, count]
reviews = []

with open('reviews.json', 'r') as f:
    for line in f:
        a = json.loads(line)
        if counts[int(a['stars'])] <= 0:
            continue
        if len(reviews) >= count * 5:
            break
        reviews.append((a['stars'], a['text']))
        counts[int(a['stars'])] -= 1

print("Reviews uploaded")



# Controls the Parameters
TOTAL = len(reviews)
TRAIN_MAX = int(TOTAL)
# TOTAL = 2000

TFIDFList = get_tfidf_list(reviews)
MAX_FEATURES = 100

tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english', ngram_range=(1, 3), max_df=1.0,
min_df=1, max_features= MAX_FEATURES, binary=False, use_idf=True,
smooth_idf=True, sublinear_tf=True)

tfidf.fit_transform(TFIDFList)

print("TFIDF completed")

########  Training the classifier

trainNew = []

with open('wekayelp.arff', 'w') as weka:
    weka.write('@relation yelpdataset\n\n')
    for i in range(0, MAX_FEATURES):
        weka.write("@attribute " + "WORD"+str(i) + " " + "real" + "\n")
    weka.write("@attribute " + "STAR" + " " + "{S1,S2,S3,S4,S5}" + "\n\n")

with open('wekayelp.arff', 'a') as weka:
    weka.write('@data\n')
    for i in range(0, TRAIN_MAX):
        sentence = reviews[i]
        tfid_feature = tfidf.transform([sentence[1]])
        weka.write(" ".join(str(item) for item in tfid_feature.toarray()[0]) + " " + "S"+str(sentence[0]) + "\n")

