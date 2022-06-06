""" Model subclasses """
import tensorflow as tf

def build_image_classifier(base_model, classes, input_shape=(512, 512, 1), classifier_activation="softmax", dropout=None):
    model = tf.keras.Sequential()

    # Load base architecture
    # Prior to TF 2.8.0, some apps will not take classifier activation param
    try:
        for layer in base_model(weights=None, input_shape=input_shape, classifier_activation=classifier_activation).layers:
            model.add(layer)
    except TypeError:
        for layer in base_model(weights=None, input_shape=input_shape).layers:
            model.add(layer)

    # Remove prediction layer
    model.pop()

    # Add dropout if desired
    if dropout:
        model.add(tf.keras.layers.Dropout(dropout))

    # Add new prediction layer
    model.add(tf.keras.layers.Dense(classes))

    return model
