import xgboost as xgb
import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
from py4j.java_gateway import Py4JJavaError

from estimator.model import Model, ModelGateway
from estimator.bridge import JVMConnection, INSTALLED_MODELS
from utility.functions import add_docstring, rename_to_xgb_param, cast_to_scala_type

XGB_DOC = xgb.sklearn.__estimator_doc + xgb.sklearn.__model_doc


@add_docstring(method_name='make', extra_doc=XGB_DOC)
class XGBClassifier(ModelGateway):
    """A utility class to create XGB model objects """

    @classmethod
    def make(cls, *, backend='python', spark=None, **kwargs):
        """
        Creates a XGB classifier object with either Python or Scala backend

        :param backend: 'python' or 'spark'
        :param spark: a SparkSession. Not needed when backend='python'
        :param kwargs: keyword parameters for the object constructor
        :return: a model object
        """
        if backend.lower() == 'python':
            return XGBClassifierPython(**kwargs)
        elif backend.lower() == 'spark':
            if cls.__name__ not in INSTALLED_MODELS:
                raise NotImplementedError(f'Scala class is not set up for {cls.__name__}')

            if not JVMConnection.is_spark_set():
                if not spark:
                    # TODO: log
                    pass
                JVMConnection.set_spark(spark)
            return XGBClassifierScala(**kwargs)
        else:
            raise Exception(f'Backend {backend.lower()} not supported. Please choose either "python" or "spark"')

    @classmethod
    def load_model(cls, model_file, *, backend='python', spark=None):
        """
        Load a model from either a file path or a byte stream
        Know caveats:
            * In python, it is permissible to load the saved booster from python or booster.saveModel() from spark
            * In spark, it can only
        :param model_file:
        :param backend:
        :param spark:
        :return:
        """
        if backend.lower() == 'python':
            clf = cls.make()
            clf.load_model(model_file)
            return clf
        elif backend.lower() == 'spark':
            if cls.__name__ not in INSTALLED_MODELS:
                raise NotImplementedError(f'Scala class is not set up for {cls.__name__}')

            if not JVMConnection.is_spark_set():
                if not spark:
                    # TODO: log
                    pass
                JVMConnection.set_spark(spark)

            clf = cls.make(backend=backend, spark=spark)
            clf.load_model(model_file)
            return clf
        else:
            raise Exception(f'Backend {backend.lower()} not supported. Please choose either "python" or "spark"')


@add_docstring(extra_doc=XGB_DOC)
@add_docstring(method_name='fit', extra_doc=xgb.XGBClassifier.fit.__doc__)
@add_docstring(method_name='predict', extra_doc=xgb.XGBModel.predict.__doc__)
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


@add_docstring(extra_doc=XGB_DOC)
class XGBClassifierScala(Model):
    """ A class represents XGB implementation in Scala"""

    def __init__(self, **kwargs):
        self.connection = JVMConnection.get_active()
        self._model = None
        self._booster = None
        self._feature_col = 'features'
        self._label_col = 'label'
        self._eval_sets = None
        self._train_objective_history = None
        self._validation_objective_history = None
        self._raw_eval_sets = kwargs.get('eval_sets', {})
        if 'eval_sets' in kwargs:
            del kwargs['eval_sets']
        xgb_param = cast_to_scala_type(rename_to_xgb_param(kwargs))
        self._clf = self.connection.jvm.XGBoostClassifier(xgb_param) \
            .setFeaturesCol(self._feature_col) \
            .setLabelCol(self._label_col)

    @property
    def eval_sets(self):
        return self._eval_sets

    @property
    def fitted_model(self):
        return self._model

    @property
    def booster(self):
        return self._booster

    @property
    def num_tress_built(self):
        """
        Sometimes it x 3. Maybe due to num_class?
        :return:
        """
        return len(list(self.booster.getModelDump("", False, "text")))

    @property
    def train_objective_history(self):
        return self._train_objective_history

    @property
    def validation_objective_history(self):
        return self._validation_objective_history

    def _extract_objective_history(self):
        if not self.fitted_model:
            return

        summary = self.fitted_model.summary()
        self._train_objective_history = list(summary.trainObjectiveHistory())
        self._validation_objective_history = {}
        java_validation = summary.validationObjectiveHistory()
        for i in range(java_validation.length()):
            item = java_validation.apply(i)
            self._validation_objective_history[item._1()] = list(item._2())

    def _on_model_update(self):
        self._booster = self._model.nativeBooster()

        try:
            self._extract_objective_history()
        except Py4JJavaError as e:
            # TODO: log
            pass

    def transform(self, labeled_data: DataFrame, features: list):
        """
        Vectorizes the pyspark dataframe so it can be consumed directly by .fit() and .predict()
        :param labeled_data:
        :param features:
        :return:
        """
        vector_assembler = VectorAssembler(
            inputCols=features,
            outputCol=self._feature_col
        )

        eval_sets = {
            key: vector_assembler.transform(df).select(self._feature_col, self._label_col)._jdf
            for key, df in self._raw_eval_sets.items()
        }
        self._eval_sets = cast_to_scala_type(eval_sets)
        self._clf.setEvalSets(self._eval_sets)

        return vector_assembler.transform(labeled_data).select(self._feature_col, self._label_col)

    def fit(self, vector_data: DataFrame):
        """
        Fits the classifier on the data and returns itself
        :param vector_data:
        :return: itself
        """
        self._model = self._clf.fit(vector_data._jdf)
        self._on_model_update()

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
        self._feature_col = col_name
        self._clf.setFeaturesCol(self._feature_col)
        return self

    def set_label_col(self, col_name: str):
        self._label_col = col_name
        self._clf.setLabelCol(self._label_col)
        return self

    def save_model(self, file_path):
        """
        Saves the Spark model to the specified location. The saved file can only be loaded by Spark XGBoost
        :param file_path: Default points to an HDFS location. Use file:/// to point to a local location
        :return:
        """
        self.fitted_model.write().overwrite().save(file_path)

    def save_booster(self, file_path):
        """
        Saves the native booster to the specified location. The saved file can be loaded by Python XGBoost but not Spark
        :param file_path:
        :return:
        """
        self.booster.saveModel(file_path)

    def load_model(self, file_path):
        """
        Loads a Spark model, not a native booster file
        :param file_path:
        :return:
        """
        self._model = self.connection.jvm.XGBoostClassificationModel.load(file_path)
        self._on_model_update()
