import json
from nltk import tokenize
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
import operator
from nltk.stem.porter import PorterStemmer


#  Set of the StopWords identified  #####

stop = set(('and', 'or', 'not', 'the', ',', '(', ')', '!', 'are', 'I', 'i', 'am', 'to', 'at', 'a', 'will', 'was', 'by',
            'in', 'of'))

# Set of the StopWords identified  #####


# Using the stemming feature or not

stemmer = PorterStemmer()
STEMMING_ON = False


complete = dict()


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


from nltk.sentiment.vader import SentimentIntensityAnalyzer
i = 0
review = []
with open('reviews.json', 'r') as f:
    for line in f:
        if i >= 200000:
            break
        a = json.loads(line)
        review.append((a['stars'], a['text']))
        i += 1

print(len(review))
TRAIN_MAX = 500
# TOTAL = 200000
TOTAL = 2000


#sid = SentimentIntensityAnalyzer()


# for i in range(0, 100):
#     sentence = review[i]
#     ss = sid.polarity_scores(sentence[1])
#     for k in sorted(ss):
#         print('{0}: {1}, '.format(k, ss[k]), end='')
#     print("")



########  Training the classifier

trainNew = []

for i in range(0, TRAIN_MAX):
    sentence = review[i]
    token_count = get_word_count(word_tokenize(sentence[1]))
    trainNew.append((token_count, sentence[0]))

classifier = nltk.classify.MaxentClassifier.train(trainNew, 'GIS', trace = 0, max_iter=100)
test = []
ans = []
test_ans = []

####### Tokenize and store in the memory

i_token = dict()

for i in range(TRAIN_MAX, TOTAL):
    sentence = review[i]
    i_token[i] = get_word_count(word_tokenize(sentence[1]))

####### Tokenize and store in the memory

print("Reached here")
for i in range(TRAIN_MAX, TOTAL):
    sentence = review[i]
    token_count = i_token[i]  # returns the TOKEN_COUNT Dictionary
    test.append(token_count)
    ans.append(sentence[0])
    test_ans.append((token_count, sentence[0]))

# i = 0
# for featureset in test:
#     #print(featureset)
#     pdist = classifier.prob_classify(featureset)
#     # print('%8.2f %6.2f %6.2f %6.2f %6.2f The answer is %i' % (pdist.prob(1), pdist.prob(2), pdist.prob(3), pdist.prob(3),pdist.prob(5), ans[i]))
#     i += 1


print("The final accuracy is " + str(nltk.classify.accuracy(classifier, test_ans)))

# sorted_x = sorted(complete.items(), key=operator.itemgetter(1), reverse = True)
# print(sorted_x)

# test = []
# ans = []
# for i in range(20, 50):
#     sentence = review[i]
#     tokens = tuple(word_tokenize(sentence[1]))
#     ans.append(sentence[0])
#     test.append(tokens)
#
#
# for t in test:
#





# '''for j in range(len(stars)):
#     f = open(os.path.join(str(stars[j]),review[j]+'.txt'), 'w',encoding='utf-8')
#     f.write(text[j])
#     f.close()'''
#
# for i,j in review:
#     print('Ratings: ' + str(j), i)