from abc import ABCMeta, abstractmethod


class Model(metaclass=ABCMeta):
    """Abstract class indicating an algorithm"""

    @abstractmethod
    def transform(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass


class ModelGateway(metaclass=ABCMeta):
    """Abstract class indicating an entry point to create models"""

    @classmethod
    @abstractmethod
    def make(cls):
        pass
