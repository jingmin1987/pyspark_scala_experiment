from model import Model, ModelGateway


class XGBClassifier(ModelGateway):
    """A utility class to create XGB model objects """

    @classmethod
    def make(cls, *args, backend='python', **kwargs):
        if backend.lower() == 'python':
            return XGBClassifierPython(*args, **kwargs)
        elif backend.lower() == 'scala':
            return XGBClassifierScala(*args, **kwargs)
        else:
            raise Exception('Backend not supported. Please choose either "python" or "scala"')


class XGBClassifierPython(Model):
    pass


class XGBClassifierScala(Model):
    pass

