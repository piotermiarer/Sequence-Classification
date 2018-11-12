# generic class must be extended
class SequenceTransformer:
    def transform(self, data):
        raise NotImplementedError

    def transform_to_predict(self, data):
        raise NotImplementedError
