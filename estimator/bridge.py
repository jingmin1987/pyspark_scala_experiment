from pyspark.sql import SparkSession


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

    @property
    def jvm(self):
        return self.spark._jvm

    @property
    def util(self):
        return self.jvm.jz.scala.util
