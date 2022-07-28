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


# TODO: need to create dummy booster/model files and dataset for testing
def test_xgbclassifier_load(connection):
    pass


def test_xgbclassifier_transform(connection):
    pass


def test_xgbclassifier_fit(connection):
    pass


def test_xgbclassifier_predict(connection):
    pass
