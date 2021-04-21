from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression

class PrecipitationTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that makes a boolean value indicating whether annual
    precipitation was "low". Mainly for example purposes rather than a
    meaningful feature engineering step.
    """
    def fit(self, X, y):
        return self

    def transform(self, X, y=None):
        X_new = X.copy()
        X_new["low_precipitation"] = [int(x < 12)
                                      for x in X_new["annual_precipitation"]]
        return X_new
