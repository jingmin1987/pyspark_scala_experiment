from pyspark.sql import SparkSession
from py4j.java_gateway import java_import


INSTALLED_MODELS = {
    'XGBClassifier': 'ml.dmlc.xgboost4j.scala.spark.XGBoostClassifier'
}


class JVMConnection:
    """
    A class manages the connection between python runtime and spark JVM
    """

    _active_connection = None

    @classmethod
    def set_spark(cls, spark: SparkSession):
        """
        Sets the global spark session for JVMConnection. Technically user should only set it once
        :param spark: A SparkSession object
        :return:
        """
        if cls._active_connection:
            # Log warning
            pass
        else:
            assert isinstance(spark, SparkSession)
            cls._active_connection = cls(spark)

    @classmethod
    def get_active(cls):
        """
        Returns the currently active JVMConnection. If no active session available, it will create one
        :return:
        """
        if not cls._active_connection:
            spark = SparkSession.builder.getOrCreate()
            cls._active_connection = cls(spark)
        return cls._active_connection

    def __init__(self, spark=None):
        """
        Initializes an instance of JVMConnection. It shouldn't be called by user
        :param spark: A SparkSession object
        """
        if not spark:
            spark = SparkSession.builder.getOrCreate()

        self.spark = spark
        for model, package in INSTALLED_MODELS.items():
            java_import(self.jvm, package)

    @property
    def jvm(self):
        return self.spark._jvm

    @property
    def util(self):
        return self.jvm.jz.scala.util.Conversion

    def cmd(self, script: str):
        """
        A utility function to run command line inside JVM. Mostly for debugging purposes
        :param script: a command such as `where is python`
        :return: None
        """
        process = self.jvm.Runtime.getRuntime().exec(script).getInputStream()
        buffer = self.jvm.io.BufferedReader(self.jvm.io.InputStreamReader(process))
        while self.jvm.System.out.println(buffer.readLine()):
            pass
