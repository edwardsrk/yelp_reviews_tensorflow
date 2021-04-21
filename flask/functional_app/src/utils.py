import pandas as pd
import nltk
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import regexp_tokenize, word_tokenize, RegexpTokenizer
nltk.download('punkt')
nltk.download('tagsets')
nltk.help.upenn_tagset()
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
import pickle #install
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
from tabulate import tabulate
import sys


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

def key_feat_producer(feats, review, prediction):
    """Takes in a list of words and a prediction"""
    pred = str(prediction)
    #print('pred',type(pred), pred, file = sys.stderr)
    #print('review',type(review), review, file = sys.stderr)
    word_list = review.split()
    #print('wordlist', word_list, file = sys.stderr)
    new_df=feats[feats['features'].isin(word_list)]
    #print('new_df', new_df, file = sys.stderr)
    sort = new_df.sort_values(by = 'values')
    #print('sort', sort, file = sys.stderr)
    pos_df = sort.tail()
    #print('posdf', pos_df, file = sys.stderr)
    neg_df = sort.head()
    #print('pred', type(pred), pred, file = sys.stderr)
    if pred == 'pos':
        #print('pos', file = sys.stderr)
        return (tabulate(pos_df, headers='keys', tablefmt='psql', showindex= False))
        #return pos_df
    else:
        #print('neg', file = sys.stderr)
        return (tabulate(neg_df, headers='keys', tablefmt='psql', showindex= False))
        #return neg_df

def extract_feature_values(data):
    """ Given a params dict, return the values for feeding into a model"""
    
    # Replace these features with the features for your model. They need to 
    # correspond with the `name` attributes of the <input> tags
    EXPECTED_FEATURES = [
        "adult_antelope_population",
        "annual_precipitation",
        "winter_severity_index"
    ]

    # This assumes all inputs will be numeric. If you have categorical features
    # that the user enters as a string, you'll want to rewrite this as a for
    # loop that treats different features differently
    values = [[float(data[feature]) for feature in EXPECTED_FEATURES]]
    return pd.DataFrame(values, columns=EXPECTED_FEATURES)
