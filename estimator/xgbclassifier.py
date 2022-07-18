import xgboost as xgb
import pandas as pd
from estimator.model import Model, ModelGateway


class XGBClassifier(ModelGateway):
    """A utility class to create XGB model objects """

    @classmethod
    def make(cls, backend='python', spark=None, **kwargs):
        """
        Creates a XGB classifier object with either Python or Scala backend

        :param backend: 'python' or 'scala'
        :param spark: a SparkSession. Not needed when backend='python'
        :param kwargs: keyword parameters for the object constructor
        :return: a model object
        """
        if backend.lower() == 'python':
            return XGBClassifierPython(**kwargs)
        elif backend.lower() == 'scala':
            return XGBClassifierScala(**kwargs)
        else:
            raise Exception('Backend not supported. Please choose either "python" or "scala"')

# TODO: keyword binding so no extra keyword is supplied
class XGBClassifierPython(xgb.XGBClassifier, Model):
    """A simple wrapper of the original XGBClassifier"""

    def __init__(self, **kwargs):
        """
        A simple wrapper of xgboost.XGBClassifier

        :param kwargs: same arguments used to instantiate xgboost.XGBClassifier
        """
        if xgb.__version__ >= "0.72":
            super().__init__(**kwargs)
        else:
            raise NotImplementedError("Not implemented yet for XGB < 0.72")

    def transform(self, x, y):
        """
        Converts the data to pd.DataFrame so it can be consumed by .fit() and predict()

        :param x: an object that can be converted to a pd.DataFrame or pd.Series
        :param y: an object that can be converted to a pd.DataFrame or pd.Series
        :return: two pandas dataframes or series
        """
        df_x, df_y = x, y
        if not isinstance(df_x, pd.DataFrame):
            df_x = pd.DataFrame(df_x)

        if not isinstance(df_y, pd.DataFrame):
            df_y = pd.DataFrame(df_y)

        return df_x, df_y

    def fit(self, x, y, **kwargs):
        """
        Fits the model on the data

        :param x: a pd.DataFrame or pd.Series
        :param y: a pd.DataFrame or pd.Series
        :param kwargs: other keyword parameters
        :return: fitted model
        """
        assert x.shape[0] == y.shape[0]
        return super().fit(x, y, **kwargs)

    def predict(self, x, **kwargs):
        """
        Predicts class based on inputs

        :param x: a pd.DataFrame or pd.Series
        :param kwargs: other keyword parameters
        :return: predicted classes with the same number of rows as x
        """
        return super().predict(x, **kwargs)


class XGBClassifierScala(Model):
    pass

