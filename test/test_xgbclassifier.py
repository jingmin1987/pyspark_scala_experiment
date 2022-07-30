import pytest

from estimator.xgbclassifier import XGBClassifier, XGBClassifierPython, XGBClassifierSpark
from test.test_util import connection


def test_xgbclassifier_make(connection):
    xgb_python = XGBClassifier.make()
    assert isinstance(xgb_python, XGBClassifierPython)

    xgb_spark = XGBClassifier.make(backend='spark', spark=connection.spark)
    assert isinstance(xgb_spark, XGBClassifierSpark)


def test_xgbclassifier_make_exception(connection):
    with pytest.raises(NotImplementedError):
        XGBClassifier.make(backend='julia')


def test_xgbclassifier_load(connection):
    python_booster = XGBClassifier.load_model('booster/python_booster.bin')
    assert isinstance(python_booster, XGBClassifierPython)

    spark_booster = XGBClassifier.load_model('booster/spark_booster.bin')
    assert isinstance(spark_booster, XGBClassifierPython)

    spark_model = XGBClassifier.load_model('booster/spark_model', backend='spark', spark=connection.spark)
    assert isinstance(spark_model, XGBClassifierSpark)
