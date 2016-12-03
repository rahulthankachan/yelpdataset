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
BAG_OF_WORDS_ON = False


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

# Generate the sentiment Score Dictionary
def get_sentiment_score_dictionary(sentence):

    sentiment = dict()
    ss = sid.polarity_scores(sentence)

    # print("The type is" + str(type(ss)) + " the len is " + str(len(ss)))

    if ss['neg'] > 0.50:
        sentiment['NEG'] = 1
    if ss['pos'] > 0.50:
        sentiment['POS'] = 1
    if ss['neu'] > 0.50:
        sentiment['NEU'] = 1

    # if ss['neg'] < 0.10:
    #     sentiment['NEG'] = 0
    # if ss['pos'] < 0.10:
    #     sentiment['POS'] = 0
    # if ss['neu'] < 0.10:
    #     sentiment['NEU'] = 0

    return sentiment

# Generate the POS tags dictionary
def get_pos_tags_dictionary(sentence):

    pos_tags = dict()
    for token, pos in pos_tag(word_tokenize(sentence)):
        pos_tags[pos] = 1
    return pos_tags

# Generate the Positive Tag Dictionary
def get_positive_tags_dictionary(sentence):

    positive = dict()
    for token in word_tokenize(sentence):
        if token in POSITIVE_WORDS:
            positive["POSITIVE_" + token] = 1
    return positive

# Generate the Negative Tag Dictionary
def get_negative_tags_dictionary(sentence):

    negative = dict()
    for token in word_tokenize(sentence):
        if token in NEGATIVE_WORDS:
            negative["NEGATIVE_" + token] = 1
    return negative


# Reading all the reviews
i = 0

# # # # # # # # # # # # # # # # # Configure the Size # # # # # # # # # # # # # #
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
TRAIN_MAX = int(TOTAL * .7)
# TOTAL = 2000

TFIDFList = get_tfidf_list(reviews)

tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english', ngram_range=(1, 3), max_df=1.0,
min_df=1, max_features= 100, binary=False, use_idf=True,
smooth_idf=True, sublinear_tf=True)

tfidf.fit_transform(TFIDFList)

print("TFIDF completed")

########  Training the classifier

trainNew = []
for i in range(0, TRAIN_MAX):
    sentence = reviews[i]

    tfid_feature = tfidf.transform([sentence[1]])
    # print(len(tfid_feature.toarray()[0]))

    # token_count = get_word_count(word_tokenize(sentence[1]))  # earlier used feature

    token_count = get_token_score(tfid_feature.toarray()[0])
    if VADER_ON:
        token_count.update(get_sentiment_score_dictionary(sentence[1]))
        #time.sleep(1)

    if POS_ON:
        temp = get_pos_tags_dictionary(sentence[1])
        if temp is not None:
            token_count.update(temp)

    if BAG_OF_WORDS_ON:
        token_count.update(get_positive_tags_dictionary(sentence[1]))
        token_count.update(get_negative_tags_dictionary(sentence[1]))


    #print(token_count)
    trainNew.append((token_count, sentence[0]))

# encoding = maxent.TypedMaxentFeatureEncoding.train(
# trainNew, count_cutoff=2, alwayson_features=True)

print("Classifier training start")
classifier = nltk.classify.MaxentClassifier.train(trainNew, algorithm='GIS', trace = 0, max_iter= 250)
# classifier = nltk.classify.MaxentClassifier.train(trainNew, encoding=encoding, trace = 0, max_iter= 3)

print("Trained the classifier")



tokens = []
tokens_ans = []
tokens_dev = []
original_labels = []  # Will store all the original labelled ratings for the reviews
predicted_labels = []

####### Tokenize and store in the memory  ## Till Here

i_token = dict()

for i in range(TRAIN_MAX, TOTAL):
    sentence = reviews[i]
    i_token[i] = get_word_count(word_tokenize(sentence[1]))


####### Tokenize and store in the memory

print("Reached here")

# # # # This is for the original - Only Tokens # #
# for i in range(TRAIN_MAX, TOTAL):
#     sentence = reviews[i]
#
#     tokens_count = i_token[i]  # returns the TOKENS_COUNT Dictionary
#
#     tokens_dev.append(tokens_count)
#     original_labels.append(sentence[0])
#     tokens_ans.append((tokens_count, sentence[0])) # this is having thr mapping of the tokens - actual label
# # # # This is for the original - Only Tokens # #


# # # # This is for the TF-IDF Sections # #
for i in range(TRAIN_MAX, TOTAL):
    sentence = reviews[i]
    tfid_feature = tfidf.transform([sentence[1]])

    #tokens_count = i_token[i]  # returns the TOKENS_COUNT Dictionary
    tokens_count = get_token_score(tfid_feature.toarray()[0])

    if VADER_ON:
        token_count.update(get_sentiment_score_dictionary(sentence[1]))

    if POS_ON:
        token_count.update(get_pos_tags_dictionary(sentence[1]))

    if BAG_OF_WORDS_ON:
        token_count.update(get_positive_tags_dictionary(sentence[1]))
        token_count.update(get_negative_tags_dictionary(sentence[1]))



    tokens_dev.append(tokens_count)
    original_labels.append(sentence[0])

    tokens_ans.append((tokens_count, sentence[0])) # this is having thr mapping of the tokens - actual label


# # # # # # Classification and Evaluation  # # # # #

i = 0
for featureset in tokens_dev:
    pdist = classifier.prob_classify(featureset)
    predict = (pdist.prob(1), pdist.prob(2), pdist.prob(3), pdist.prob(3), pdist.prob(5), original_labels[i])

    max_val = pdist.prob(1)
    max_index = 1
    for index in range(1, 6):
        if pdist.prob(index) >= max_val:
            max_val = pdist.prob(index)
            max_index = index

    predicted_labels.append(max_index)
    #print('%8.2f %6.2f %6.2f %6.2f %6.2f The answer is %i' % (pdist.prob(1), pdist.prob(2), pdist.prob(3), pdist.prob(4),pdist.prob(5), original_labels[i]))
    i += 1


# print("The final accuracy is " + str(nltk.classify.accuracy(classifier, tokens_ans)))

print("Len of the first is " + str(len(original_labels)) + " len of the other is " + str(len(predicted_labels)))
print(precision_recall_fscore_support(original_labels,predicted_labels,average='weighted'))
