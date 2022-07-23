import xgboost as xgb
import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
from estimator.model import Model, ModelGateway
from estimator.bridge import JVMConnection, INSTALLED_MODELS


class XGBClassifier(ModelGateway):
    """A utility class to create XGB model objects """

    @classmethod
    def make(cls, *, backend='python', spark=None, **kwargs):
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
            if cls.__name__ not in INSTALLED_MODELS:
                raise NotImplementedError(f'No Scala package installed for {cls.__name__}')

            if not JVMConnection.is_spark_set():
                if not spark:
                    # Log warning
                    pass
                JVMConnection.set_spark(spark)
            return XGBClassifierScala(**kwargs)
        else:
            raise Exception(f'Backend {backend.lower()} not supported. Please choose either "python" or "scala"')


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
        if not isinstance(df_x, (pd.DataFrame, pd.Series)):
            df_x = pd.DataFrame(df_x)

        if not isinstance(df_y, (pd.DataFrame, pd.Series)):
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
    """ A class represents XGB implementation in Scala"""

    def __init__(self, **kwargs):
        self.connection = JVMConnection.get_active()
        xgb_param = self.connection.util.toMap(kwargs)
        self._clf = self.connection.jvm.XGBoostClassifier(xgb_param)
        self._model = None

    def transform(self, labeled_data: DataFrame, features: list, label: str):
        """
        Vectorizes the pyspark dataframe so it can be consumed directly by .fit() and .predict()
        :param labeled_data:
        :param features:
        :param label:
        :return:
        """
        vector_assembler = VectorAssembler(
            inputCols=features,
            outputCol='features'
        )
        return vector_assembler.transform(labeled_data).select('features', label)

    def fit(self, vector_data: DataFrame):
        """
        Fits the classifer on the data and returns itself
        :param vector_data:
        :return: itself
        """
        self._model = self._clf.fit(vector_data._jdf)
        return self

    def predict(self, vector_data: DataFrame):
        """
        Batch prediction. Returns a pyspark dataframe
        :param vector_data:
        :return:
        """
        java_df = self._model.transform(vector_data._jdf)
        return DataFrame(java_df, self.connection.spark)

    def set_features_col(self, col_name: str):
        self._clf.setFeaturesCol(col_name)
        return self

    def set_label_col(self, col_name: str):
        self._clf.setLabelCol(col_name)
        return self
