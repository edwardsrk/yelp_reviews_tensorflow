import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
nltk.download('punkt')
nltk.download('tagsets')
nltk.help.upenn_tagset()
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import regexp_tokenize, word_tokenize, RegexpTokenizer

import random

import pickle #install

import pandas as pd

from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn import metrics

from statistics import mode #install

import json

import csv

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

from os import path

from PIL import Image

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from xgboost import XGBClassifier

def get_wordnet_pos(treebank_tag):
    '''
    Translate nltk POS to wordnet tags
    '''
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def data_cleaner(doc):
    """A function to strip punctuation, strip stopwords, casefold, lemmatize,
    And part pf speech tag words for clean data for modeling"""
    
    sw = stopwords.words('english')
    regex_token = RegexpTokenizer(r"([a-zA-Z]+(?:’[a-z]+)?)")
    doc = regex_token.tokenize(doc)
    doc = [word.lower() for word in doc]
    doc = [word for word in doc if word not in sw]
    #print(doc)
    doc = pos_tag(doc)
    doc = [(word[0], get_wordnet_pos(word[1])) for word in doc]
    #print(doc)
    lemmatizer = WordNetLemmatizer() 
    doc = [lemmatizer.lemmatize(word[0], word[1]) for word in doc]
    #print(' '.join(doc))
    return ' '.join(doc)

def num_to_cat(star):
    """Gives data a classification tag based on its star rating"""
    if star == 4 or star == 5:
        return 'pos'
    else:
        return 'neg'

def conf_matrix_plotter(model, X_t_vec, y_t):
    """create confusion matrix plots"""
    fig, ax = plt.subplots()

    fig.suptitle(str(model))

    plot_confusion_matrix(model, X_t_vec, y_t, ax=ax, cmap="plasma");
    
    
def wordcloud_maker(df, stopwords = None):
    """cretes words clouds from cleaned data"""
    all_clean = " ".join(review for review in df.clean)
    wordcloud = WordCloud(stopwords = stopwords, background_color="white").generate(all_clean)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    
def gb_cleaner(df):
    """Fixes tags, cleans the text and drops unneccessary columns, data is ready to be put through model"""
    df['tag'] = df.tags.apply(retagger)
    
    c_list = df.text.tolist()

    clean_corpus = []
    for docs in c_list:
        clean_corpus.append(data_cleaner(docs))
    
    df['clean'] = clean_corpus

    df = df.drop(['text', 'tags', 'stars'], axis= 1)
    
    return df

def retagger(tags):
    """Standardizes tag column to look be either 'pos' or 'neg' and is based off of grubhub sentiment analysis"""
    if tags == 'Positive':
        return 'pos'
    else:
        return 'neg'
    
def feature_finder(model):
    """Takes in a model and pulls out the most relevant 5000 features
        and their coefficients. Returns a dataframe of features and coefficients"""
    
    features = model.steps[0][1].get_feature_names()
    feat_values = model[1].coef_

    c = {'features' : features}
    feats = pd.DataFrame(data = c)
    feats['values'] = feat_values[0]

    sorted_feats = feats.sort_values(by='values')
    return  sorted_feats

def key_feat_producer(review, prediction,feats):
    """Takes in a list of words and a prediction"""
    pred = list(prediction)
    word_list = data_cleaner(review).split()
    new_df=feats[feats['features'].isin(word_list)]
    sort = new_df.sort_values(by = 'values')
    if prediction[0] == 'pos':
        return sort.tail()
    else:
        return sort.head()
    
def single_review_prep(text):
    """Takes in a single review as a string of text, 
    cleans the review and puts it into a form the model can predict on"""
    clean_test = data_cleaner(text)
    dummy_dict= {'star': [clean_test]}
    clean_test_df = pd.DataFrame(dummy_dict)
    return clean_test_df
