from abc import ABCMeta, abstractmethod


class Model(metaclass=ABCMeta):
    """Abstract class indicating an algorithm"""

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass


class ModelGateway(metaclass=ABCMeta):
    """Abstract class indicating an entry point to create models"""

    @classmethod
    @abstractmethod
    def make(cls):
        pass
