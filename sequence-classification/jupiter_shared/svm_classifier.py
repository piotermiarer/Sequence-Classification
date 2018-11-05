from sequence_classifier import SequenceClassifier
from sequence_transformer import SequenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

from sklearn.svm import SVC


class SVMClassifier(SequenceClassifier):
    def __init__(self, name='SVM'):
        super(SVMClassifier, self).__init__(name)
        self.model = SVC(gamma='auto')

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def get_transformer(self):
        return SVMTransformer()


class SVMTransformer(SequenceTransformer):
    def __init__(self):
        self.vectorizer = CountVectorizer()

    def transform(self, raw_data):
        # count vectorizer
        X = self.vectorizer.fit_transform(raw_data)
        return X.toarray()

