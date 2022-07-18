import xgboost as xgb
import pandas as pd
from estimator.model import Model, ModelGateway


class XGBClassifier(ModelGateway):
    """A utility class to create XGB model objects """

    @classmethod
    def make(cls, backend='python', **kwargs):
        if backend.lower() == 'python':
            return XGBClassifierPython(**kwargs)
        elif backend.lower() == 'scala':
            return XGBClassifierScala(**kwargs)
        else:
            raise Exception('Backend not supported. Please choose either "python" or "scala"')


class XGBClassifierPython(xgb.XGBClassifier, Model):
    """A simple wrapper of the original XGBClassifier"""
    def __init__(self, **kwargs):
        if xgb.__version__ >= "0.72":
            super().__init__(**kwargs)
        else:
            raise NotImplementedError("Not implemented yet for XGB < 0.72")

    def transform(self, x, y):
        df_x, df_y = x, y
        if not isinstance(df_x, pd.DataFrame):
            df_x = pd.DataFrame(df_x)

        if not isinstance(df_y, pd.DataFrame):
            df_y = pd.DataFrame(df_y)

        return df_x, df_y

    def fit(self, x, y, **kwargs):
        return super().fit(x, y, **kwargs)

    def predict(self, x, **kwargs):
        return super().predict(x, **kwargs)


class XGBClassifierScala(Model):
    pass

