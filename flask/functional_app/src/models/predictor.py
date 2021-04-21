import pickle
import pandas as pd


def un_pickle_model():
    """ Load the model from the .pkl file """
    with open ('C:/Users/edwardsrk/final_proj/yelp_reviews_tensorflow/data/pickles/final_sgd_model', "rb") as model_p: 
        loaded_model = pickle.load(model_p)
    return loaded_model

def get_prediction(feature_values):
    """ Given a list of feature values, return a prediction made by the model"""
    print(feature_values)
    loaded_model = un_pickle_model()
    
    # Model is expecting a list of lists, and returns a list of predictions
    predictions = loaded_model.predict(feature_values)
    # We are only making a single prediction, so return the 0-th value
    return predictions[0]

def feature_finder():
    """Takes in a model and pulls out the most relevant 5000 features
        and their coefficients. Returns a dataframe of features and coefficients"""
    loaded_model = un_pickle_model()
    features = loaded_model.steps[0][1].get_feature_names()
    feat_values = loaded_model[1].coef_

    c = {'features' : features}
    feats = pd.DataFrame(data = c)
    feats['values'] = feat_values[0]

    sorted_feats = feats.sort_values(by='values')
    return  sorted_feats
