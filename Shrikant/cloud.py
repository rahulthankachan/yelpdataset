import json
import pickle
import os
import pymongo
import nltk
from nltk import re, wordpunct_tokenize, defaultdict, NaiveBayesClassifier
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from os import path
from wordcloud import WordCloud
#wordcloud = WordCloud().generate(text)
import matplotlib.pyplot as plt
data = []
i = 0
reviews = []

labels = []
features = []
fivestars = dict()
fourstars = dict()
threestars = dict()
twostars = dict()
onestar = dict()
stars = []
count5=0
count4=0
count3=0
count2=0
count1=0
toker = nltk.RegexpTokenizer(r'\w+')
from nltk.corpus import stopwords
stop = set(stopwords.words('English'))
with open('C:\yelp_academic_dataset_review.json') as f:
    for line in f:
        if i >= 100000:
            break
        a = json.loads(line)
        if a['stars'] == 5:
            wordset = toker.tokenize(a['text'])
            for word in wordset:
                if str.lower(word) in stop:
                   continue
                features.append(str.lower(word))
        i += 1
        '''elif a['stars'] == 2 or a['stars'] == 1:
            reviews.append((wordpunct_tokenize(a['text']), 'negative'))
            labels.append('negative')
        else:
            reviews.append((wordpunct_tokenize(a['text']), 'neutral'))
            labels.append('neutral')
        features.append(wordpunct_tokenize(a['text']))
        i += 1
        #stars.append(a['stars'])
        if a['stars'] == 5:
            count5 +=1
        elif a['stars'] ==4:
            count4 +=1
        elif a['stars'] ==3:
            count3 +=1
        elif a['stars'] ==2:
            count2 +=1
        else:
            count1 += 1'''

from nltk import FreqDist
list = FreqDist(features)
print(list.most_common(300))
f = open('1star.txt','w', encoding='utf-8')
s = ""
removewords = set(['food','service','time','like','get','take','manager','day','come','say','car','know','would','get','told','said','good','went','restaurant','minutes','came','make','ordered','people','got','go','another','customer','asked','place','order','even','minute','going','hour','called'])
count1=0
for i,j in list.most_common(300000):
    count1 += 1
    if count1 < 100:
        continue
    #if i in removewords:
    #    continue
    for k in range(j):
        f.write(i + " ")


f.close()
file = open('1star.txt','r',encoding='utf-8')
text = file.readlines()
print(text)

# lower max_font_size
wordcloud = WordCloud(max_font_size=40).generate(str(text))
print(wordcloud)
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
plt.savefig('figure1.jpeg')
f.close()
'''
print(count1)
print(count2)
print(count3)
print(count4)
print(count5)
#for i in range(1,6):
c.writerow(str(count5),)
c.writerow(str(count4),)
c.writerow(str(count3),)
c.writerow(str(count2),)
c.writerow(str(count1),)
f.close()'''

