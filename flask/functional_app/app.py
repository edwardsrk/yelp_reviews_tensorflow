from flask import Flask, send_from_directory, render_template, request, redirect, url_for
from waitress import serve
from src.utils import data_cleaner, get_wordnet_pos, key_feat_producer
from src.models.predictor import get_prediction, feature_finder
import sys
import pandas as pd



app = Flask(__name__, static_url_path="/static")

@app.route("/")
def index():
    """Return the main page."""
    return send_from_directory("static", "index.html")

@app.route("/make_prediction", methods=["POST"])
def make_prediction():
    """ Use the ML model to make a prediction using the form inputs. """
    #print('hello', file = sys.stderr)
   # Get the data from the submitted form
    data = request.form
    
    clean_r = data_cleaner(data.get('user_review_prediction'))
    #print(clean_r, file = sys.stderr)
    raw_dict= {'review' : [clean_r]}
    clean_df = pd.DataFrame(raw_dict)
    #print('clean f df was printed here', clean_df, file = sys.stderr)

    # Convert the data into just a list of values to be sent to the model
    #feature_values = extract_feature_values(data)
    #print(feature_values) # Remove this when you're done debugging

    # Send the values to the model to get a prediction
    prediction = get_prediction(clean_df.review)
    # Tell the browser to fetch the results page, passing along the prediction
    #print('hello', file = sys.stderr)
    feats = feature_finder()
    #print(feats, file = sys.stderr)
    
    #print(feats, file = sys.stderr)
    #print(type(feats), file = sys.stderr)
    
    #print(clean_r, file = sys.stderr)
    #print(type(clean_r), file = sys.stderr)
    
    #print(prediction, file = sys.stderr)
    #print(type(prediction), file = sys.stderr)
    
    key_feats = key_feat_producer(feats, clean_r, prediction)
    print('key feats', key_feats, file = sys.stderr)
    #print('made it below both functs', file = sys.stderr)
    key_feats = (key_feats.replace('\n', '<br>'))
    
    
    return redirect(url_for("show_results", prediction=prediction, key_feats=key_feats))

@app.route("/show_results")
def show_results():
    """ Display the results page with the provided prediction """
    
    # Extract the prediction from the URL params
    if request.args.get("prediction") == 'pos':
        prediction = 'Positive'
    else:
        prediction = 'Negative'
    #prediction = request.args.get("prediction")
    key_feats = request.args.get("key_feats")

    # Round it for display purposes
    #prediction = (prediction)

    # Return the results pge
    return render_template("results.html", prediction=prediction, key_feats= key_feats)


if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=5000)
