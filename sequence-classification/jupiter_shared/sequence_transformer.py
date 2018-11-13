# generic class must be extended
class SequenceTransformer:
    def fit_transform(self, data):
        raise NotImplementedError

    def transform(self, data):
        raise NotImplementedError
