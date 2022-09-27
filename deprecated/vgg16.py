""" VGG-16 Keras implementation """
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense


def VGG_16(input_shape=(224,224,3), weightsPath=None, classes=1000, dropout=0.5, classifier_activation='softmax'):
    model = Sequential()

    model.add(Conv2D(input_shape=input_shape, filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='conv1_1'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='conv1_2'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', name='conv2_1'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', name='conv2_2'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name='conv3_1'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name='conv3_2'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name='conv3_3'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv4_1'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv4_2'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv4_3'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv5_1'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv5_2'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name='conv5_3'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Flatten(name='flatten'))
    model.add(Dense(units=4096, activation='relu', name='dense_1'))
    model.add(Dropout(dropout))
    model.add(Dense(units=4096, activation='relu', name='dense_2'))
    model.add(Dropout(dropout))
    model.add(Dense(units=classes, name='dense_3'))
    model.add(Activation(classifier_activation, name=classifier_activation))

    if weightsPath:
        model.load_weights(weightsPath)

    return model
