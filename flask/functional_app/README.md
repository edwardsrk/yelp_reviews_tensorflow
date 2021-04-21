# Capstone Flask App Template

## Getting Started with the Existing Web App

Before attempting to modify this template to use your model, make sure you have the starter app up and running.  It is very difficult to debug whether the problem is the code or the installation/setup if you jump right in and modify the code before testing out the starter app.

### Requirements

This repo uses Python 3.8.5, Flask 1.1.2, and scikit-learn 0.23.2. All python packages can be found in the `requirements.txt` file.  The requirements are in `pip` style, because this is supported by Heroku.

To create a new `conda` environment to use this repo, run:
```bash
conda create --name flask-env pip
conda activate flask-env
pip install -r requirements.txt
```

Note that this environment does not include Jupyter Notebook, it only includes the requirements for the Flask app.

### Running the Flask Application

To run in a development environment (on your local computer), run:
```bash
export FLASK_ENV=development
env FLASK_APP=app.py flask run
```

This will produce an output that looks something like:
```
 * Serving Flask app "app.py" (lazy loading)
 * Environment: development
 * Debug mode: on
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: <PIN>
```

Like Jupyter Notebook, this server needs to stay running in the terminal for the application to work.  If you want to do something else in the terminal, you need to open a new window/tab, or shut down the server with CTRL+C. **DO NOT** just close the terminal window when you are done running the Flask app — it will keep running in the background and cause problems until you locate the process ID and terminate it — always make sure you use CTRL-C.

Unlike Jupyter Notebook, this doesn't open in the browser automatically.  You need to copy the URL http://127.0.0.1:5000/ and paste it into a web browser.  Once you do that, you should see the homepage of the example app!

## How to Use This Template

### Requirements

You will likely need to install additional packages to support your deployment if you are using packages other than just scikit-learn (and its dependencies).  With the `flask-env` activated, you can run `pip install <package-name>` (e.g. `pip install tensorflow`).

Some students prefer to use "error-driven development" here, meaning they add code then see what error messages come up, rather than trying to figure out exactly what packages they need in advance.

**Once your web app is working and you are ready to deploy**, you can generate your own `requirements.txt` for reproducibility purposes with:
```bash
pip freeze > requirements.txt
```

This will overwrite the current file in this repository with one that includes your new requirements.

### Input Form: HTML

The input form is located under `static/index.html`.  Go through and replace all elements of the template app with relevant content for your project.  There are comments throughout to guide you in what you replace.

We recommend that you replace the content first, then work on restyling later if you have time. The general structure of this form should work with most apps.

For image classification tasks, we recommend that you display a few example images on the screen and allow the user to choose which one they want a prediction on with a drop-down.  Uploading custom images (or having the user provide a URL to a custom image) is possible, it just may be challenging given your time constraints in Capstone.

Overall, make sure you replace the following in `static/index.html`:

 - [ ] The `<title>` (which appears in the browser tab)
 - [ ] The `<h1>` (what we might think of as the "title" more conventionally — the large text at the top of the page)
 - [ ] The `<img>` (image + caption)
 - [ ] The `<label>`s and `<input>`s in the form (except for the final submit button input)
 - [ ] The `<p>`s (paragraphs at the bottom)

### Features: Python

The code that converts the HTML form data into features for your model is in `src/utils.py`. Replace the list of features from the sample app with your own features

### Model (Pipeline): Python

Prior to deploying with a Flask app, you need a pickled model. (In theory you could fit a model every time you made a prediction, but it's much faster and better practice to fit the model once, then use that fitted model to make a prediction.)  See [this repo](https://github.com/learn-co-students/capstone-model-pickling-082420) for an example of how to pickle a model.

Replace the pickled model (`src/models/model.pkl`) with your model file.  If you did not pickle a pipeline, you will also need to modify the code in `src/models/predictor.py` so it performs the necessary preprocessing steps on the data from the web form.

If there are any custom classes needed by your code, make sure you copy them into the `models` submodule, in a separate file that is imported.  You can go ahead and delete the `PrecipitationTransformer` from the example if you want to reuse `src/models/custom_transformers.py`.

### Results: HTML

Once you are able to submit data through the form and make a prediction without the app crashing (yay!), customize the results page HTML. 

Take a look at:

 - [ ] The `<title>`
 - [ ] The text describing the result
 - [ ] The `<img>`

## Live Deployment

You can definitely just use the web app for tech demo day, but ideally you're eventually able to publish it live so you can just send the URL to friends, family, and prospective employers.

We recommend using Heroku for this purpose, since it is fairly straightforward to use.  For some advanced or very large models (e.g. Spark) it may cost money, but typical scikit-learn models should be able to deploy for free.

To run in a production environment (used for deployment, but test it out locally first):
```bash
export FLASK_ENV=production
python app.py
```

If that code works, export your new `requirements.txt`, add and commit your changes, and push to GitHub.  Now all you need to do is make an account on [Heroku](https://signup.heroku.com/), provide a link to your GitHub repo, and deploy! There is a file called `Procfile` that tells Heroku how to run your app.
