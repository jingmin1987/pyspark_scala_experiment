from pyspark.sql import SparkSession
from py4j.java_gateway import java_import


INSTALLED_MODELS = {
    'XGBClassifier': 'ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier'
}


class JVMConnection:

    _active_connection = None

    @classmethod
    def set_spark(cls, spark):
        if cls._active_connection:
            # Log warning
            pass
        else:
            assert isinstance(spark, SparkSession)
            cls._active_connection = cls(spark)

    @classmethod
    def get_active(cls):
        if not cls._active_connection:
            spark = SparkSession.builder.getOrCreate()
            cls._active_connection = cls(spark)
        return cls._active_connection

    def __init__(self, spark=None):
        if not spark:
            spark = SparkSession.builder.getOrCreate()

        self.spark = spark
        for model, package in INSTALLED_MODELS.items():
            java_import(self.spark, package)

    @property
    def jvm(self):
        return self.spark._jvm

    @property
    def util(self):
        return self.jvm.jz.scala.util.Conversion
