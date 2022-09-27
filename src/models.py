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

# UNet Keras implementation
def unet(
        metrics, # No default value; must be defined as part of the MirroredStrategy to use multiple GPUs
        input_height=512,
        input_width=512,
        input_channels=1, # Grayscale = 1, color = 3 for RGB
        num_classes=1,
        dropout=0.5,
        filters=64,
        output_activation='sigmoid', # 'sigmoid' or 'softmax'
        weights=None, # Load weights for model from file
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.99),
        num_layers=4):
    # Create input from dimensions
    input_shape = (input_height, input_width, input_channels)
    inputs = tf.keras.layers.Input(input_shape)
    x = inputs

    down_layers = []
    for l in range(num_layers):
        x = conv2d_block(inputs=x, filters=filters)
        down_layers.append(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), strides=2) (x)
        filters = filters * 2 # double the number of filters with each layer

    x = tf.keras.layers.Dropout(dropout)(x)
    x = conv2d_block(inputs=x, filters=filters)

    for conv in reversed(down_layers):
        filters //= 2 # decreasing number of filters with each layer
        x = tf.keras.layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
        ch, cw = get_crop_shape(tf.keras.backend.int_shape(conv), tf.keras.backend.int_shape(x))
        conv = tf.keras.layers.Cropping2D(cropping=(ch, cw))(conv)
        x = tf.keras.layers.Concatenate()([x, conv])
        x = conv2d_block(inputs=x, filters=filters)

    outputs = tf.keras.layers.Conv2D(num_classes, (1, 1), activation=output_activation)(x)
    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])

    if weights:
        model.load_weights(weights)

    return model


def conv2d_block(
        inputs,
        use_batch_norm=False,
        dropout=0.0,
        filters=64,
        kernel_size=(3,3),
        activation='relu',
        kernel_initializer='he_normal',
        padding='same'):
    c = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=not use_batch_norm
    )(inputs)

    if use_batch_norm:
        c = tf.keras.layers.BatchNormalization()(c)

    if dropout > 0.0:
        c = tf.keras.layers.Dropout(dropout)(c)

    c = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        activation=activation,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=not use_batch_norm
    )(c)

    if use_batch_norm:
        c = tf.keras.layers.BatchNormalization()(c)

    return c


def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = target[2] - refer[2]
    assert cw >= 0
    if cw % 2 != 0:
        cw1, cw2 = int(cw/2), int(cw/2) + 1
    else:
        cw1, cw2 = int(cw/2), int(cw/2)

    # height, the 2nd dimension
    ch = target[1] - refer[1]
    assert ch >= 0
    if ch % 2 != 0:
        ch1, ch2 = int(ch/2), int(ch/2) + 1
    else:
        ch1, ch2 = int(ch/2), int(ch/2)

    return (ch1, ch2), (cw1, cw2)
