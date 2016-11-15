import json
from nltk import word_tokenize
import nltk
from nltk.classify import NaiveBayesClassifier

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


for i in range(0, 100):
    temp = review[i]
    tokens = word_tokenize(temp[1])
    text = nltk.Text(tokens)
    print(text.collocations())








# '''for j in range(len(stars)):
#     f = open(os.path.join(str(stars[j]),review[j]+'.txt'), 'w',encoding='utf-8')
#     f.write(text[j])
#     f.close()'''
#
# for i,j in review:
#     print('Ratings: ' + str(j), i)