from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from sequence_classifier import SequenceClassifier
from sequence_transformer import SequenceTransformer


class NeuralNetworksClassifier(SequenceClassifier):
    def __init__(self, name='Neural Networks', top_words=5000, max_review_length=500):
        super(NeuralNetworksClassifier, self).__init__(name)
        self.max_review_length = max_review_length
        embedding_vecor_length = 32
        self.model = Sequential()
        self.model.add(Embedding(top_words, embedding_vecor_length, input_length=self.max_review_length))
        self.model.add(LSTM(100))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=3, batch_size=64)

    def predict(self, X):
        return self.model.predict(X)

    def get_transformer(self):
        return NeuralNetworksTransformer(self.max_review_length)


class NeuralNetworksTransformer(SequenceTransformer):
    def __init__(self, max_review_length=500):
        self.max_review_length = max_review_length

    def transform(self, X):
        # truncate and pad input sequences
        return sequence.pad_sequences(X, maxlen=self.max_review_length)

    def transform_to_predict(self, X):
        return sequence.pad_sequences(X, maxlen=self.max_review_length)

