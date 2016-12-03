import json
import math
import re
import nltk.sentiment
import string

from sklearn import tree
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.classification import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import cross_val_score
from nltk.stem.porter import *
from nltk.tokenize import *

#Total data count to run the models. 
TOTAL = 200000

#Train data count selected from total data.
TRAINMAX = 140000
#Vader Creation
C = nltk.sentiment.vader.SentimentIntensityAnalyzer(); 

#Stemmer Creation 
stemmer = PorterStemmer() 

# List to raw text for TF-IDF to train on
TFIDFList = []

# Negative words list creation
negetiveWords = []
with open('negetiveWords.txt') as n:
    lines = n.read();
    for line in lines.split('\n'):
        negetiveWords.append(line)

# Positive words list creation        
positiveWords = []
with open('positiveWords.txt') as p:
    lines = p.read();
    for line in lines.split('\n'):
        positiveWords.append(line)
      
#Method to perform the stemming 
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

#Method to tokenize the string
def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

#Mehtod to calculate pos tag count for main pos tags
def getPosCount(posz):
    RB = 0
    VB = 0
    NN = 0
    JJ = 0
    for i in range(len(posz)):
        if(posz[i][1] == 'RB' or posz[i][1] == 'RBR' or posz[i][1] == 'RBS'):
            RB+=1
        elif(posz[i][1] == 'VB' or posz[i][1] == 'VBD' or posz[i][1] == 'VBG' or posz[i][1] == 'VBN' or posz[i][1] == 'VBP'):
            VB+=1
        elif(posz[i][1] == 'NN' or posz[i][1] == 'NNP' or posz[i][1] == 'NNS' or posz[i][1] == 'NNPS'):
            NN+=1
        elif(posz[i][1] == 'JJ' or posz[i][1] == 'JJR' or posz[i][1] == 'JJS'):
            JJ+=1
            
    return [RB, VB, NN, JJ]   
     

#method to create raw text set for TF - IDF training.
def extractReviews(reviews):
    for i in range(0, TRAINMAX):
        lowers = reviews[i][1].lower()
#         print(lowers)
        no_punctuation = lowers.translate(string.punctuation)
        TFIDFList.append(no_punctuation)    
    
    
    
        
i = 0  

#List containing all the reviews     
reviews = []

#Count for containing review counts of particular type
Star1Count = 0
Star2Count = 0
Star3Count = 0
Star4Count = 0
Star5Count = 0

#Total review count
totalCount = 0

#opening of json file containing reviews
with open('yelp_academic_dataset_review.json') as f:
    for line in f:
        if totalCount >=TOTAL:
            break;
        a = json.loads(line)
        if(a['stars'] == 1 and Star1Count < TOTAL/5):
            Star1Count+=1;
            reviews.append((a['stars'], a['text']))
            totalCount+=1
        elif(a['stars'] == 2 and Star2Count < TOTAL/5):
            Star2Count+=1;
            reviews.append((a['stars'], a['text']))
            totalCount+=1
        elif(a['stars'] == 3 and Star3Count < TOTAL/5):
            Star3Count+=1;
            reviews.append((a['stars'], a['text']))
            totalCount+=1
        elif(a['stars'] == 4 and Star4Count < TOTAL/5):
            Star4Count+=1;
            reviews.append((a['stars'], a['text']))
            totalCount+=1 
        elif(a['stars'] == 5 and Star5Count < TOTAL/5):
            Star5Count+=1;
            reviews.append((a['stars'], a['text']))
            totalCount+=1 

#Loading of TFIDFList with text of training data
extractReviews(reviews)

# Creation of TfidfVectorizer along with various parameter tunings
tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english', ngram_range=(1, 3), max_df=1.0,
min_df=1, max_features=100, binary=False, use_idf=True,
smooth_idf=True, sublinear_tf=True)

#Training of TF-IDF model and calculating most frequent words along with there frequencies
tfidf.fit_transform(TFIDFList)

X = []
Y =[]
for review in reviews[:TRAINMAX]:
    reviewList = []
#These below values were used to hold probability calculation of bag-of-words in each review    
#     prob1Final = 0
#     prob2Final = 0
#     prob3Final = 0
#     prob4Final = 0
#     prob5Final = 0
    negetiveScore = 0
    positiveScore = 0
    pos = nltk.pos_tag(review[1])
    posList = getPosCount(pos)
    tokens = tokenize(review[1])
    tokens = stem_tokens(tokens, stemmer)

    for token in tokens:
#         token = re.sub(r'\W+', '', token)
##These below values were used in probability calculation of bag-of-words in each review           
#         if token in wordValue.keys():
#             prob1Final+=math.log(wordValue[token][0])
#             prob2Final+=math.log(wordValue[token][1])
#             prob3Final+=math.log(wordValue[token][2])
#             prob4Final+=math.log(wordValue[token][3])
#             prob5Final+=math.log(wordValue[token][4])

#Counting negative words
        if token.lower() in negetiveWords:
            negetiveScore+=1

#Counting positive words            
        if token.lower() in positiveWords:
            positiveScore+=1
            

#Calculating Vader sentiment values for each review            
    v = C.polarity_scores(review[1])
#     List to hold values of particular review to be in particular Class   
#     valueList = [prob1Final, prob2Final, prob3Final, prob4Final, prob5Final]

#Selecting the maximum probability and using that as a feature
#     minProbClass = valueList.index(max(valueList))

#Using tfidf model on this particular review
    tfsResult = tfidf.transform([review[1]])

#reviewList will contain feature set of a particular review
    reviewList = [v['neu'], v['compound'], v['pos'], v['neg']]
    reviewList.extend([negetiveScore, positiveScore])
    reviewList.extend(posList) 
    reviewList.extend(tfsResult.toarray()[0])  
    
#List containing feature set of all the reviews        
    X.append(reviewList)
    
#List containing labels for each review    
    Y.append(review[0])

#Decision Tree classifier creation
clf = tree.DecisionTreeClassifier(criterion='entropy')

#Classifier With parameter tunings
# clf = tree.DecisionTreeClassifier(criterion='entropy', splitter="random", presort = True, min_impurity_split = 1e-5, max_depth = 50)

#Training of classifier created above
clf = clf.fit(X, Y)


#Cross - validation code with KFold
'''
scores = cross_val_score(clf,     # Model to test
                X,  
                Y,      # Target variable
                scoring = "accuracy",               # Scoring metric    
                cv=3)

print (scores)

'''
# Lists for containing test feature set and labels
X1 = []
Y1 = []
for review in reviews[TRAINMAX:TOTAL]:
    reviewList = []
#     prob1Final = 0
#     prob2Final = 0
#     prob3Final = 0
#     prob4Final = 0
#     prob5Final = 0
    negetiveScore = 0
    positiveScore = 0
    pos = nltk.pos_tag(review[1])
    posList = getPosCount(pos)
    tokens = tokenize(review[1])
    tokens = stem_tokens(tokens, stemmer)
#     print(tokens)
    for token in tokens:
        token = re.sub(r'\W+', '', token)
#         if token in wordValue.keys():
#             prob1Final+=math.log(wordValue[token][0])
#             prob2Final+=math.log(wordValue[token][1])
#             prob3Final+=math.log(wordValue[token][2])
#             prob4Final+=math.log(wordValue[token][3])
#             prob5Final+=math.log(wordValue[token][4])
        if token.lower() in negetiveWords:
            negetiveScore+=1
            
        if token.lower() in positiveWords:
            positiveScore+=1
            
    v = C.polarity_scores(review[1])
#     valueList = [prob1Final, prob2Final, prob3Final, prob4Final, prob5Final]
#     minProbClass = valueList.index(max(valueList))
    tfsResult = tfidf.transform([review[1]])
    
    reviewList = [v['neu'], v['compound'], v['pos'], v['neg']] 
    reviewList.extend([negetiveScore, positiveScore])
    reviewList.extend(posList)
    reviewList.extend(tfsResult.toarray()[0])         
    
    X1.append(reviewList)
    Y1.append(review[0])

# Predicting on unseen test data
Y2 = clf.predict(X1);

# Calculating precision, recall ,F1 - Score and Accuracy and printing them 
print(precision_recall_fscore_support(Y1,Y2,average='weighted'))

accuracy = clf.score(X1, Y1)

#Printing Accuracy
print(accuracy)

 
