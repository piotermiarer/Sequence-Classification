from sklearn.metrics import accuracy_score


class SequenceClassifierComparator:
    def __init__(self, classifiers=None):
        if classifiers is None:
            classifiers = []
        self.classifiers = classifiers
        self.predictions = []
        self.scores = []

    def add_classifier(self, classifier, sequence_transformer=None):
        self.classifiers.append((classifier, sequence_transformer))

    def fit_classifiers(self, X_train, y_train):
        for classifier, transformer in self.classifiers:
            if transformer is not None:
                classifier.fit(transformer.transform(X_train), y_train)
            else:
                classifier.fit(X_train, y_train)

    def predict_classifiers(self, X):
        self.predictions = []
        for classifier, transformer in self.classifiers:
            if transformer is not None:
                y_pred = classifier.predict(transformer.transform_to_predict(X))
            else:
                y_pred = classifier.predict(X)
            self.predictions.append((classifier.name, y_pred))
        return self.predictions

    def score_classifiers(self, X_test, y_test):
        for name, y_pred in self.predict_classifiers(X_test):
            y_pred = [0 if i < 0.5 else 1 for i in y_pred]
            accuracy = accuracy_score(y_test, y_pred)
            self.scores.append((name, accuracy))
        return self.scores

    def plot_comparison(self):
        raise NotImplementedError
