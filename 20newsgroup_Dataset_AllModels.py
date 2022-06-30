#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 00:14:02 2020

@author: nithila
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from scipy.sparse import hstack
from sklearn_pandas import DataFrameMapper

import re 
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
import string as str
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV, KFold

from wordcloud import WordCloud, STOPWORDS


from sklearn.preprocessing import MinMaxScaler, Normalizer
import time
start_time = time.time()
scaler = MinMaxScaler()
pd.set_option('display.max_colwidth', -1)

'''Importing Train data from the 20 newsgroup data'''
train_data_1 = fetch_20newsgroups(subset = 'train', shuffle = True, random_state = 7, remove=(['headers','footers','quotes']))

(train_data_1.target_names)
len( train_data_1.target)
print(list(train_data_1.target_names))

target_values, freq = np.unique(train_data_1.target, return_index = False, return_counts = True)
print(target_values,freq)
labels = target_values
#labels = train_data_1.target_names
sizes = freq

fig1, ax1 = plt.subplots()
ax1.pie(sizes,labels=labels, shadow=True, startangle=90, autopct='%1.f%%')
ax1.axis('equal') 
#plt.title('Newsgroup Train Class Splits')
#plt.legend(labels,loc=1)

plt.show()

''' Storing train data into a dataframe to further create features out of it'''
'''this provess will then be repeated for both the datasets and their corresponding test train components'''

train_data_1_df = pd.DataFrame({'raw_data':train_data_1.data, 'act_class':train_data_1.target})
#print(train_data_1_df.iloc[0:10])

'''Basic Metrics'''

train_data_1_df['no_of_words'] = train_data_1_df['raw_data'].apply(lambda a: sum([i.strip(str.punctuation).isalpha() for i in a.split()]))
train_data_1_df['no_of_letters'] = train_data_1_df['raw_data'].str.len()
train_data_1_df['average'] = train_data_1_df['no_of_letters']/train_data_1_df['no_of_words']

train_data_1_df['average'] = train_data_1_df['average'].replace([np.nan, np.inf], 0)


''' Cleaning '''

train_data_1_df['raw_data'] = train_data_1_df['raw_data'].str.replace('\d+', ' ') #Removed numbers
train_data_1_df['raw_data'] = train_data_1_df['raw_data'].str.replace('\W+', ' ')
train_data_1_df['raw_data'] = train_data_1_df['raw_data'].str.lower()


''' Tokenize'''
train_data_1_df['raw_data_token'] = train_data_1_df['raw_data'].apply(nltk.word_tokenize)


''' Lemmatization '''
lemmatize_words = WordNetLemmatizer()
train_data_1_df['lemmed_words'] = train_data_1_df['raw_data_token'].apply(lambda a: [lemmatize_words.lemmatize(b) for b in a])
#train_data_1_df = train_data_1_df.drop(columns=['raw_data'])


''' Stemming '''
stem_words = SnowballStemmer("english")
train_data_1_df['lem_stem_words'] = train_data_1_df['lemmed_words'].apply(lambda a: [stem_words.stem(b) for b in a])
train_data_1_df = train_data_1_df.drop(columns=['lem_stem_words'])


''' POS tagging '''
train_data_1_df['tagged_words'] = train_data_1_df['lemmed_words']. apply(lambda a: nltk.pos_tag(a))

def NounCount(x):
   data = []
   for (w, p) in x:
        if p.startswith("NN"):
            data.append(w)
   return len(data)

train_data_1_df['nouns'] = train_data_1_df['tagged_words'].apply(NounCount)


def ProNounCount(x):
   data = []
   for (w, p) in x:
        if p in ['PRP','PRP$','WP','WP$']:
            data.append(w)
   return len(data)

train_data_1_df['pronouns'] = train_data_1_df['tagged_words'].apply(ProNounCount)


def VerbCount(x):
   data = []
   for (w, p) in x:
        if p in ['VB','VBD','VBG','VBN','VBP','VBZ']:
            data.append(w)
   return len(data)

train_data_1_df['verbs'] = train_data_1_df['tagged_words'].apply(VerbCount)



def AdVerbCount(x):
   data = []
   for (w, p) in x:
        if p in ['RB','RBR','RBS','WRB']:
            data.append(w)
   return len(data)

train_data_1_df['adverbs'] = train_data_1_df['tagged_words'].apply(AdVerbCount)



def AdjectiveCount(x):
   data = []
   for (w, p) in x:
        if p in ['JJ','JJR','JJS']:
            data.append(w)
   return len(data)

train_data_1_df['adjectives'] = train_data_1_df['tagged_words'].apply(AdjectiveCount)

''' WORDCLOUD '''

stp = set(STOPWORDS)
trn_words = train_data_1_df[train_data_1_df['act_class']==19]

trn_words_array = ' '.join(trn_words['raw_data'].tolist())

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stp, 
                min_font_size = 10).generate(trn_words_array) 

# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 
  


train_data_1_df.groupby(
   ['act_class']
).agg(
    {
         'nouns':"mean",    
         'verbs': "mean",  
         'adjectives': "mean",
         'no_of_words': "mean",
         'act_class':"count"
    }
)



'''



vect = TfidfVectorizer(ngram_range=(1, 2), binary =True, stop_words='english',analyzer='word', min_df=2, preprocessor = ' '.join) 
train_features1_1 = vect.fit_transform(train_data_1_df['lemmed_words'])
#features_train = vect.get_feature_names()

#len(features_train)
#print(train_features1_1.shape)


train_target_1 = train_data_1_df['act_class']
train_target_1 = train_target_1.tolist()


#train_features1_1_df = pd.DataFrame.sparse.from_spmatrix(train_features1_1, columns = vect.get_feature_names())

to_be_added_train_df = train_data_1_df
to_be_added_train_df[['no_of_words','no_of_letters','average','nouns','pronouns','verbs','adjectives','adverbs']] = scaler.fit_transform(to_be_added_train_df[['no_of_words','no_of_letters','average','nouns','pronouns','verbs','adjectives','adverbs']])  
to_be_added_train_df.drop(['raw_data', 'act_class','raw_data_token','tagged_words'], axis=1, inplace=True)

to_be_added_train_arr1 = to_be_added_train_df
to_be_added_train_csr = sparse.csr_matrix(to_be_added_train_arr1)


train_features_1_data = hstack([to_be_added_train_csr,train_features1_1])

'''


'''**********'''

''' FINAL TRAIN SET - TARGET '''
train_target_1 = train_data_1_df['act_class']
train_target_1 = train_target_1.tolist()



train_data_1_df[['no_of_words','no_of_letters','average','nouns','pronouns','verbs','adjectives','adverbs']] = scaler.fit_transform(train_data_1_df[['no_of_words','no_of_letters','average','nouns','pronouns','verbs','adjectives','adverbs']])  
train_data_1_df.drop(['raw_data', 'act_class','raw_data_token','tagged_words','act_class'], axis=1, inplace=True)

vectorize = DataFrameMapper([
     ('lemmed_words', TfidfVectorizer(ngram_range=(1, 2), binary =True, stop_words='english',strip_accents= 'ascii',lowercase = True ,analyzer='word', max_features=20000, preprocessor = ' '.join)),
     (['no_of_words'], None),
     (['no_of_letters'], None),
     (['average'], None),
     (['nouns'], None),
     (['pronouns'], None),
     (['verbs'], None),
     (['adverbs'], None),
     (['adjectives'], None),
     
 ])

''' FINAL TRAIN SET - FEATURES'''    
    
train_features_1 = vectorize.fit_transform(train_data_1_df)








'''***********************************************************'''
'''***********************************************************'''
'''***********************************************************'''

''' PRE PROCESSING FOR TRAIN DATA FOR DATASET 1 ENDS '''

'''***********************************************************'''
'''***********************************************************'''
'''***********************************************************'''


''' Importing Test data for the 20 newsgroup data'''

test_data_1 = fetch_20newsgroups(subset = 'test', shuffle = True, random_state = 7, remove=(['headers','footers','quotes']))

len(test_data_1.target_names)
len( test_data_1.target)
print(list(test_data_1.target_names))

a=[0,1,2,3,4,5,6,7,8,9,10]

for iter in a:
    print(test_data_1.data[iter], '\n',test_data_1.target[iter],'\n', test_data_1.target_names[iter])

target_values, freq = np.unique(test_data_1.target, return_index = False, return_counts = True)
print(target_values,freq)
labels = target_values
#labels = test_data_1.target_names
sizes = freq

fig1, ax1 = plt.subplots()
ax1.pie(sizes,labels=labels, shadow=True, startangle=90, autopct='%1.f%%')
ax1.axis('equal') 
#plt.title('Newsgroup test Class Splits')
#plt.legend(labels,loc=1)

plt.show()

''' Storing test data into a dataframe to further create features out of it'''
'''this provess will then be repeated for both the datasets and their corresponding test test components'''

test_data_1_df = pd.DataFrame({'raw_data':test_data_1.data, 'act_class':test_data_1.target})
#print(test_data_1_df.iloc[0:10])

'''Basic Metrics'''

test_data_1_df['no_of_words'] = test_data_1_df['raw_data'].apply(lambda a: sum([i.strip(str.punctuation).isalpha() for i in a.split()]))
test_data_1_df['no_of_letters'] = test_data_1_df['raw_data'].str.len()
test_data_1_df['average'] = test_data_1_df['no_of_letters']/test_data_1_df['no_of_words']

test_data_1_df['average'] = test_data_1_df['average'].replace([np.nan, np.inf], 0)

''' Cleaning '''

test_data_1_df['raw_data'] = test_data_1_df['raw_data'].str.replace('d+', ' ') #Removed numbers
test_data_1_df['raw_data'] = test_data_1_df['raw_data'].str.replace('W+', ' ')
test_data_1_df['raw_data'] = test_data_1_df['raw_data'].str.lower()

test_data_1_df['raw_data'] = test_data_1_df['raw_data'].map(lambda x: re.sub(r'\W+', ' ', x))


''' Tokenize'''
test_data_1_df['raw_data_token'] = test_data_1_df['raw_data'].apply(nltk.word_tokenize)


''' Lemmatization '''
lemmatize_words = WordNetLemmatizer()
test_data_1_df['lemmed_words'] = test_data_1_df['raw_data_token'].apply(lambda a: [lemmatize_words.lemmatize(b) for b in a])
#test_data_1_df = test_data_1_df.drop(columns=['raw_data'])


''' Stemming '''
stem_words = SnowballStemmer("english")
test_data_1_df['lem_stem_words'] = test_data_1_df['lemmed_words'].apply(lambda a: [stem_words.stem(b) for b in a])
test_data_1_df = test_data_1_df.drop(columns=['lem_stem_words'])

''' POS tagging '''
test_data_1_df['tagged_words'] = test_data_1_df['lemmed_words'].apply(lambda a: nltk.pos_tag(a))



test_data_1_df['nouns'] = test_data_1_df['tagged_words'].apply(NounCount)



test_data_1_df['pronouns'] = test_data_1_df['tagged_words'].apply(ProNounCount)



test_data_1_df['verbs'] = test_data_1_df['tagged_words'].apply(VerbCount)



test_data_1_df['adverbs'] = test_data_1_df['tagged_words'].apply(AdVerbCount)



test_data_1_df['adjectives'] = test_data_1_df['tagged_words'].apply(AdjectiveCount)

''' WORDCLOUD '''


stp = set(STOPWORDS)
test_words = test_data_1_df[test_data_1_df['act_class']==1]

test_words_array = ' '.join(test_words['raw_data'].tolist())

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stp, 
                min_font_size = 10).generate(test_words_array) 

# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 
  

test_data_1_df.groupby(
   ['act_class']
).agg(
    {
         'nouns':"mean",    
         'verbs': "mean",  
         'adjectives': "mean",
         'no_of_words': "mean",
         'act_class':"count"
    }
)













'''




#vect = TfidfVectorizer(ngram_range=(1, 2), binary =True, stop_words='english',analyzer='word', min_df=5) 
test_features1_1 = vect.transform(test_data_1_df['lemmed_words'])
#features_test = vect.get_feature_names()

#len(features_test)
#print(test_features1_1.shape)


test_target_1 = test_data_1_df['act_class']
test_target_1 = test_target_1.tolist()



test_features1_1_df = pd.DataFrame.sparse.from_spmatrix(test_features1_1, columns = vect.get_feature_names())

to_be_added_test_df = test_data_1_df
to_be_added_test_df.drop(['raw_data', 'act_class','raw_data_token','tagged_words'], axis=1, inplace=True)




test_features_1_df = pd.concat([test_features1_1_df, to_be_added_test_df], axis=1)

test_features_1_df[['no_of_words','no_of_letters','average','nouns','pronouns','verbs','adjectives','adverbs']] = scaler.fit_transform(test_features_1_df[['no_of_words','no_of_letters','average','nouns','pronouns','verbs','adjectives','adverbs']])



'''


''' FINAL test SET - TARGET '''
test_target_1 = test_data_1_df['act_class']
test_target_1 = test_target_1.tolist()



test_data_1_df[['no_of_words','no_of_letters','average','nouns','pronouns','verbs','adjectives','adverbs']] = scaler.fit_transform(test_data_1_df[['no_of_words','no_of_letters','average','nouns','pronouns','verbs','adjectives','adverbs']])  
test_data_1_df.drop(['raw_data', 'act_class','raw_data_token','tagged_words','act_class'], axis=1, inplace=True)
'''
vectorize = DataFrameMapper([
     ('lemmed_words', TfidfVectorizer(ngram_range=(1, 2), binary =True, stop_words='english',analyzer='word', min_df=5, preprocessor = ' '.join)),
     ('no_of_words', None),
     ('no_of_letters', None),
     ('average', None),
     ('nouns', None),
     ('pronouns', None),
     ('verbs', None),
     ('adverbs', None),
     ('adjectives', None),
     
 ])
    '''

''' FINAL test SET - FEATURES'''    
    
test_features_1 = vectorize.transform(test_data_1_df)











'''***********************************************************'''
'''***********************************************************'''
'''***********************************************************'''

''' PRE PROCESSING FOR TEST DATA FOR DATASET 1 ENDS '''

'''***********************************************************'''
'''***********************************************************'''
'''***********************************************************'''







'''***********************************************************'''
'''***********************************************************'''
'''***********************************************************'''

''' MODELLING FOR DATASET 1 '''

'''***********************************************************'''
'''***********************************************************'''
'''***********************************************************'''


#test_target_1 = np.array(test_target_1s.values.tolist())


''' LOGISTIC REGRESSION '''

pen = ['l2']
c = [0.01, 0.1, 1, 5]
hp = dict(C=c)

LogistRegr = LogisticRegression()
CV_LogistRegr = GridSearchCV(LogistRegr, hp, cv=3, n_jobs = -1)

CV_LogistRegr.fit(train_features_1, train_target_1)
#prediction1 = CV_LogistRegr.predict(test_features_1)
print("Tuned Logistic Regression Parameters: {}".format(CV_LogistRegr.best_params_)) 
print("Best score is {}".format(CV_LogistRegr.best_score_))

LogistRegrFinal = LogisticRegression(C = 1, n_jobs = -1, max_iter = 500)
LogistRegrFinal.fit(train_features_1, train_target_1)
ac = LogistRegrFinal.score(test_features_1,test_target_1)
print(ac)
#print(time.time() - start_time)





''' DECISION TREES '''

depth = [200]
hp2 = dict(max_depth=depth)

DTC = DecisionTreeClassifier()
CV_DTC = GridSearchCV(DTC, hp2, cv=3, n_jobs = -1)


CV_DTC.fit(train_features_1, train_target_1)
#prediction1 = CV_LogistRegr.predict(test_features_1)
print("Tuned Decision Tree Parameters: {}".format(CV_DTC.best_params_)) 
print("Best score is {}".format(CV_DTC.best_score_))

DTCFinal = DecisionTreeClassifier(max_depth = 200)
DTCFinal.fit(train_features_1, train_target_1)
ac = DTCFinal.score(test_features_1,test_target_1)
print(ac)



''' SUPPORT VECTOR MACHINES '''

c = [0.001, 0.01, 0.1, 1, 10, 100,2,5,15,25,50]

hp3 = dict(C = c)

SVM = LinearSVC()
CV_SVM = GridSearchCV(SVM, hp3, cv=3, n_jobs = -1)
CV_SVM.fit(train_features_1, train_target_1)

print("Tuned SVM Parameters: {}".format(CV_SVM.best_params_)) 
print("Best score is {}".format(CV_SVM.best_score_))
print(CV_SVM.best_estimator_)


SVMFinal = LinearSVC(C = 1)
SVMFinal.fit(train_features_1, train_target_1)
ac = SVMFinal.score(test_features_1,test_target_1)
print(ac)


''' ADA BOOST '''
n_est = [100,150,200]
lr=[0.5,0.8,1]
hp4 = dict(n_estimators = n_est, learning_rate = lr)

ADA = AdaBoostClassifier()
CV_ADA = GridSearchCV(ADA, hp4, cv=3, n_jobs = -1)
CV_ADA.fit(train_features_1, train_target_1)

print("Tuned ADA Parameters: {}".format(CV_ADA.best_params_)) 
print("Best score is {}".format(CV_ADA.best_score_))
print(CV_ADA.best_estimator_)


ADAFinal = AdaBoostClassifier(n_estimators = 150, learning_rate = 0.5)
ADAFinal.fit(train_features_1, train_target_1)
ac = ADAFinal.score(test_features_1,test_target_1)
print(ac)



''' RANDOM FOREST '''

n_est = [200,300,400]
depth = [50,60,90]

hp5 = dict(n_estimators = n_est, max_depth = depth)

RF = RandomForestClassifier(n_jobs = -1)
CV_RF = GridSearchCV(RF, hp5, cv=3, n_jobs = -1)
CV_RF.fit(train_features_1, train_target_1)

print("Tuned RF Parameters: {}".format(CV_RF.best_params_)) 
print("Best score is {}".format(CV_RF.best_score_))
print(CV_RF.best_estimator_)


RFFinal = RandomForestClassifier(n_estimators = 300, max_depth = 90)
RFFinal.fit(train_features_1, train_target_1)
ac = RFFinal.score(test_features_1,test_target_1)
print(ac)



''' MULTINOMIAL NB '''
a = [0.01, 0.1,1,5,2]
fp = [True, False]
cp = ['array-like',None]
hp6 = dict(alpha=a, fit_prior=fp, class_prior=cp)

MNB = MultinomialNB()
CV_MNB = GridSearchCV(MNB, hp6, cv=3, n_jobs = -1)
CV_MNB.fit(train_features_1, train_target_1)


print("Tuned RF Parameters: {}".format(CV_MNB.best_params_)) 
print("Best score is {}".format(CV_MNB.best_score_))
print(CV_MNB.best_estimator_)

MNBFinal = MultinomialNB(alpha=0.1, class_prior=None, fit_prior=False)

MNBFinal.fit(train_features_1, train_target_1)
ac = MNBFinal.score(test_features_1,test_target_1)
print(ac)