from collections.abc import Iterable

from estimator.bridge import JVMConnection


def add_docstring(method_name=None, extra_doc=''):
    """
    Decorator factory to add docstring from a different source
    :param method_name:
    :param extra_doc:
    :return:
    """
    def decorator(cls):
        if method_name:
            method = getattr(cls, method_name)
            if type(method).__name__ == 'method':
                doc = getattr(method.__func__, '__doc__', '')
                doc = doc or ''
                doc += extra_doc
                method.__func__.__doc__ = doc
            elif type(method).__name__ == 'function':
                doc = getattr(method, '__doc__', '')
                doc = doc or ''
                doc += extra_doc
                method.__doc__ = doc
        else:
            doc = getattr(cls, '__doc__', '')
            doc = doc or ''
            doc += extra_doc
            cls.__doc__ = doc
        return cls
    return decorator


def rename_to_xgb_param(sklearn_param):
    """
    Helper function to remap params between sklearn api and xgb api, INPLACE
    :param sklearn_param:
    :return:
    """
    rename_dict = {
        'n_estimators': 'num_round',
        'n_jobs': 'num_workers',
        'learning_rate': 'eta',
        'reg_alpha': 'alpha',
        'reg_lambda': 'lambda',
        'n_class': 'num_class',
        'early_stopping_rounds': 'num_early_stopping_rounds'
    }

    for key in rename_dict:
        if key in sklearn_param:
            new_key = rename_dict[key]
            sklearn_param[new_key] = sklearn_param[key]
            del sklearn_param[key]

    return sklearn_param


def cast_to_scala_type(item):
    """
    Helper function to cast python/py4j types to scala types
    :param item:
    :return:
    """
    c = JVMConnection.get_active()

    if not isinstance(item, Iterable):
        return item
    else:
        if isinstance(item, (list, tuple)):
            new_item = [cast_to_scala_type(i) for i in item]
            return c.util.toSeq(new_item)
        elif isinstance(item, dict):
            new_item = {
                key: cast_to_scala_type(val) for key, val in item.items()
            }
            return c.util.toMap(new_item)
        elif isinstance(item, set):
            new_item = {cast_to_scala_type(i) for i in item}
            return c.util.toSet(new_item)
        else:
            return item
