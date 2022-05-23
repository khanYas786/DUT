#!/usr/bin/env python
# coding: utf-8

# Loading modelling dataset
import pandas as pd
import numpy as np
import pandas_bokeh

pandas_bokeh.output_notebook()

Excel_dir = 'C://Directory/Datasets/2022 Combined Fake News/'
df = pd.read_csv(Excel_dir+'Text-label-xl-cleaned.csv')

# First lightgbm experiment

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
vectorizer = HashingVectorizer(n_features=2**15)

#tfidf  = TfidfVectorizer()

df_csvLP = pd.read_csv(Excel_dir+'Text-label-xl-cleaned.csv')

df_csvLP = df_csvLP.dropna()

X = df_csvLP['Text']
train = vectorizer.fit_transform(X)
#train = tfidf.fit_transform(X)
train = train.astype('float32')
train = train.toarray()
y = df_csvLP['Label']



X_train, X_test, y_train, y_test = train_test_split(train, y,test_size=.25,random_state =123)
#clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
clf = LGBMClassifier(learning_rate=0.09,max_depth=-5,random_state=42, n_estimators = 500)
#model = lgb.LGBMClassifier(learning_rate=0.09,max_depth=-5,random_state=42)
models = clf.fit(X_train,y_train,eval_set=[(X_test,y_test),(X_train,y_train)],
          verbose=20,eval_metric='logloss')
#models = clf.fit(X_train, X_test, y_train, y_test)
models

print('Training accuracy {:.4f}'.format(clf.score(X_train,y_train)))
print('Testing accuracy {:.4f}'.format(clf.score(X_test,y_test)))

from sklearn import metrics
metrics.plot_confusion_matrix(clf,X_test,y_test,cmap='Blues_r')

print(metrics.classification_report(y_test,clf.predict(X_test)))
import lightgbm as lgb
lgb.plot_metric(clf)

# Setting up pre-processing

import nltk.corpus
nltk.download('stopwords')
from nltk.corpus import stopwords
df = df_csvLP
stop_words = stopwords.words('english')
df['clean_title'] = df['Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))


import spacy
spacy.load('en_core_web_sm')
from spacy.lang.en import English
parser = English()
def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens



import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
from nltk.stem.wordnet import WordNetLemmatizer
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    tokens = [get_lemma2(token) for token in tokens]
    return ' '.join(tokens)


def get_topics(line):
    text_data = []
    #with open('dataset.csv') as f:
    #for line in f:
    words = prepare_text_for_lda(line)
    #print(tokens)
    text_data.append(words)
    #print(text_data)
    return ' '.join(text_data)

df['topics'] = df['clean_title'].apply(lambda x: get_topics(x))

df['topics'].str.split(expand=True).stack().value_counts()

# lightgbm model expirement post pre-processing
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
vectorizer = HashingVectorizer(n_features=2**15)

#tfidf  = TfidfVectorizer()


X = df['topics']
train = vectorizer.fit_transform(X)
#train = tfidf.fit_transform(X)
train = train.astype('float32')
train = train.toarray()
y = df['Label']


X_train, X_test, y_train, y_test = train_test_split(train, y,test_size=.25,random_state =123)
#clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
clf = LGBMClassifier(learning_rate=0.09,max_depth=-5,random_state=42, n_estimators = 440)
#model = lgb.LGBMClassifier(learning_rate=0.09,max_depth=-5,random_state=42)
models = clf.fit(X_train,y_train,eval_set=[(X_test,y_test),(X_train,y_train)],
          verbose=20,eval_metric='logloss')
#models = clf.fit(X_train, X_test, y_train, y_test)
models


print('Training accuracy {:.4f}'.format(clf.score(X_train,y_train)))
print('Testing accuracy {:.4f}'.format(clf.score(X_test,y_test)))
from sklearn import metrics
metrics.plot_confusion_matrix(clf,X_test,y_test,cmap='Blues_r')
print(metrics.classification_report(y_test,clf.predict(X_test)))
import lightgbm as lgb
lgb.plot_metric(clf)

# Refining classication outcome process
def pred_text(sentence):
    if len(sentence.split()) > 2:
        sentence = vectorizer.fit_transform([sentence])
        sentence = sentence.astype('float32')
        sentence = sentence.toarray()
        pred = clf.predict_proba(sentence)
        
    else:
        pred = [[0,1]]
    if pred[0][0] >= pred[0][1]:
        if pred[0][0] >= 0.8:
            output = "Fake"
        else:
            output = "Not Fake"
    else:
        if pred[0][0] < pred[0][1]:
            output = "Not Fake"
    return output

#Further testing of model
df_test = df[['Text', 'Label']]


df_test_true = df_test[(df_test['Label']==1)]
df_test_fake = df_test[(df_test['Label']==0)]

df_test_true['Text_Clean'] = df_test_true['Text'].apply(lambda x: get_topics(x.lower()))
df_test_true['Prediction'] = df_test_true['Text_Clean'].apply(lambda x: pred_text(x))
df_test_true.head()
df_test_true['Prediction'].value_counts().plot_bokeh(kind='bar');

df_test_fake['Text_Clean'] = df_test_fake['Text'].apply(lambda x: get_topics(x.lower()))
df_test_fake['Prediction'] = df_test_fake['Text_Clean'].apply(lambda x: pred_text(x))
df_test_fake.head()
df_test_fake['Prediction'].value_counts().plot_bokeh(kind='bar');


# Apply model on full SA Covid dataset
SA_COVID_df= pd.read_csv('C://Directory/Datasets/Cleaned-Merged.csv')
SA_COVID_df = SA_COVID_df[['date','content','id']]

SA_COVID_df['content'] = SA_COVID_df['content'].astype('str')
SA_COVID_df['Text_Clean'] = SA_COVID_df['content'].apply(lambda x: get_topics(x.lower()))
SA_COVID_df['Prediction'] = SA_COVID_df['Text_Clean'].apply(lambda x: pred_text(x))
SA_COVID_df.head()
SA_COVID_df['Prediction'].value_counts().plot_bokeh(kind='bar')

# Store data with predictions
SA_COVID_df.to_excel('C://Directory/Datasets/SA_COVID_Cleaned-Pred.xlsx', engine='openpyxl')




