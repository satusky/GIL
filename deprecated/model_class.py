""" Model subclasses """
import tensorflow as tf

class ImageClassifier(tf.keras.Model):
    def __init__(self, base_model, classes, input_shape=(512, 512, 1), classifier_activation="softmax", dropout=0.1):
        super(ImageClassifier, self).__init__()
        self.input = tf.keras.Input(shape=input_shape)
        self.base_model = base_model(include_top=False, weights=None, input_shape=input_shape, classifier_activation=classifier_activation)
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.dense = tf.keras.layers.Dense(classes)

    def call(self, inputs):
        x = self.base_model(inputs)
        x = self.dropout(x)
        x = self.dense(x)
        return x
