import pytest
from pyspark.sql import SparkSession
from estimator.bridge import JVMConnection

from util.functions import add_docstring, rename_to_xgb_param, cast_to_scala_type


@pytest.fixture()
def connection():
    spark = SparkSession.builder.appName('my_test').config('spark.jars', 'jar/scala-util.jar').getOrCreate()
    JVMConnection.set_spark(spark=spark)
    return JVMConnection.get_active()


def test_add_docstring_to_class():
    test_doc = 'test'

    @add_docstring(extra_doc=test_doc)
    class MyTest:
        pass

    assert MyTest.__doc__ == 'test'


def test_add_docstring_to_method():
    test_doc = 'test'

    @add_docstring(method_name='make', extra_doc=test_doc)
    class MyTestWithMethod:
        def make(self):
            pass

    test_class = MyTestWithMethod()
    assert test_class.make.__doc__ == test_doc


def test_rename_to_xgb_param():
    python_dict = {
        'n_estimators': 500,
        'n_jobs': 3,
        'learning_rate': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'n_class': 2,
        'early_stopping_rounds': 25
    }

    python_dict_copy = python_dict.copy()
    spark_dict = rename_to_xgb_param(python_dict)
    assert spark_dict['num_round'] == python_dict_copy['n_estimators']
    assert spark_dict['num_workers'] == python_dict_copy['n_jobs']
    assert spark_dict['eta'] == python_dict_copy['learning_rate']
    assert spark_dict['alpha'] == python_dict_copy['reg_alpha']
    assert spark_dict['lambda'] == python_dict_copy['reg_lambda']
    assert spark_dict['num_class'] == python_dict_copy['n_class']
    assert spark_dict['num_early_stopping_rounds'] == python_dict_copy['early_stopping_rounds']

    assert 'n_estimators' not in spark_dict
    assert 'n_jobs' not in spark_dict
    assert 'learning_rate' not in spark_dict
    assert 'reg_alpha' not in spark_dict
    assert 'reg_lambda' not in spark_dict
    assert 'n_class' not in spark_dict
    assert 'early_stopping_rounds' not in spark_dict


def test_cast_to_scala_type(connection):
    # I don't have a good way to test scala_list is a Seq
    python_list = [1, 2, 3]
    scala_seq = cast_to_scala_type(python_list)
    for i in range(len(python_list)):
        assert scala_seq.apply(i) == python_list[i]

    python_tuple = (1, 2, 3)
    scala_seq = cast_to_scala_type(python_tuple)
    for i in range(len(python_tuple)):
        assert scala_seq.apply(i) == python_tuple[i]

    python_dict = {'one': 1}
    scala_map = cast_to_scala_type(python_dict)
    assert scala_map.apply('one') == python_dict['one']
    assert scala_map.toString() == 'Map(one -> 1)'

    python_set = {1, 2, 1}
    scala_set = cast_to_scala_type(python_set)
    assert scala_set.toString() == 'Set(1, 2)'
