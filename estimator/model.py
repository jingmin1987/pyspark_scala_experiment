from abc import ABCMeta, abstractmethod


class Model(metaclass=ABCMeta):
    """Abstract class indicating an algorithm"""

    @abstractmethod
    def transform(self, x, y):
        pass

    @abstractmethod
    def fit(self, x, y, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, x, **kwargs):
        pass


class ModelGateway(metaclass=ABCMeta):
    """Abstract class indicating an entry point to create models"""

    @classmethod
    @abstractmethod
    def make(cls):
        pass
