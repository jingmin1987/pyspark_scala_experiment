from py4j.java_gateway import java_import
from pyspark.sql import SparkSession


class JVMConnection:
    _active_jvm = None

    @classmethod
    def get_active(cls):
        if not cls._active_jvm:
            spark = SparkSession.builder.getOrCreate()
            cls._active_jvm = spark._jvm
        return cls._active_jvm
