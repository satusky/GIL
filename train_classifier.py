""" Train VGG on image data """
import os
from datetime import datetime
import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models import build_image_classifier
from src.utility import EpochTimeCallback, get_max_batch_size
from src.data_generator import ImageSet

def model_config():
    """ Parse arguments and pull selected Keras application """
    # Available Keras application models
    model_dict = {
        "densenet121": tf.keras.applications.DenseNet121,
        "densenet169": tf.keras.applications.DenseNet169,
        "densenet201": tf.keras.applications.DenseNet201,
        "efficientnetb0": tf.keras.applications.EfficientNetB0,
        "efficientnetb1": tf.keras.applications.EfficientNetB1,
        "efficientnetb2": tf.keras.applications.EfficientNetB2,
        "efficientnetb3": tf.keras.applications.EfficientNetB3,
        "efficientnetb4": tf.keras.applications.EfficientNetB4,
        "efficientnetb5": tf.keras.applications.EfficientNetB5,
        "efficientnetb6": tf.keras.applications.EfficientNetB6,
        "efficientnetb7": tf.keras.applications.EfficientNetB7,
        "inceptionresnetv2": tf.keras.applications.InceptionResNetV2,
        "inceptionv3": tf.keras.applications.InceptionV3,
        "mobilenet": tf.keras.applications.MobileNet,
        "mobilenetv2": tf.keras.applications.MobileNetV2,
        "nasnetlarge": tf.keras.applications.NASNetLarge,
        "nasnetmobile": tf.keras.applications.NASNetMobile,
        "resnet101": tf.keras.applications.ResNet101,
        "resnet101v2": tf.keras.applications.ResNet101V2,
        "resnet152": tf.keras.applications.ResNet152,
        "resnet152v2": tf.keras.applications.ResNet152V2,
        "resnet50": tf.keras.applications.ResNet50,
        "resnet50v2": tf.keras.applications.ResNet50V2,
        "xception": tf.keras.applications.Xception
    }

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Directory containing images")
    parser.add_argument("--data_csv", required=True, metavar="CSV FILE", help="CSV file pointing to images")
    parser.add_argument("--image_column", required=True, help="Column name for images")
    parser.add_argument("--label_column", required=True, help="Column name for labels")
    parser.add_argument("--arch", required=True, choices=model_dict.keys(), help="Model architecture. Supports most Keras applications.")
    parser.add_argument("--test_ratio", help="Percentage for testing data. Default is 0.3 (30%)", type=float, default=0.3)
    parser.add_argument("--epochs", help="Number of epochs. Default is 15", type=int, default=15)
    parser.add_argument("--classes", help="Number of classes. If not specified, classes will be inferred from labels", type=int, default=None)
    parser.add_argument("--batch_size", help="Training batch size. Default is 8", type=int, default=8)
    parser.add_argument("--image_height", help="Height of images (pixels)", type=int, default=512)
    parser.add_argument("--image_width", help="Width of images (pixels)", type=int, default=512)
    parser.add_argument("--image_channels", help="Number of channels", type=int, default=1)
    parser.add_argument("--output", help="Specify file name for output. Default is 'model'", default='model')
    parser.add_argument("--auto_resize", help="Auto-resize to min height/width of image set", action="store_true")
    parser.add_argument("--auto_batch", help="Auto-detect max batch size. Selecting this will override any specified batch size", action="store_true")
    parser.add_argument('--index_first', help="Set images to depth as the first index (uncommon)", action="store_true")
    parser.add_argument("--run_eagerly", help="Run eagerly (for debug). Will lose performance.", action="store_true")
    parser.add_argument("--cross_dev_ops", help="Cross device operation to use for multi-GPU reduction. 'all' = NcclAllReduce, 'hierarchical' = HierarchicalCopyAllReduce, 'one' = ReductionToOneDevice", type=str, choices=["all", "hierarchical", "one"], default="all")
    parser.add_argument("--instance_type", help="Instance type for logging purposes", type=str, default=None)
    args = parser.parse_args()

    if args.arch not in model_dict:
        model_arch = None
    else:
        model_arch = model_dict[args.arch]

    return args, model_arch

def split_and_resize(images, labels, test_ratio, input_shape=(512, 512, 1), auto_resize=False, index_first=False, log=None):
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=test_ratio, random_state=42)

    training_set = ImageSet(images=train_images, labels=train_labels, mode="classify", index_first=index_first)
    validation_set = ImageSet(images=test_images, labels=test_labels, mode="classify", index_first=index_first)

    min_height = min([training_set.min_height, validation_set.min_height])
    min_width = min([training_set.min_width, validation_set.min_width])

    # Set input image shape
    if auto_resize:
        input_shape = (min_height, min_width, 1) # (height, width, channels)

    training_set.input_shape = input_shape
    validation_set.input_shape = input_shape

    if log:
        log.write(f"Training images: {training_set.count}\n")
        log.write(f"Validation images: {validation_set.count}\n")
        log.write(f"Min height: {min_height}\n")
        log.write(f"Min width: {min_width}\n")
        log.write(f"Input shape: {input_shape}\n\n")

    print(f"Training images: {training_set.count}")
    print(f"Validation images: {validation_set.count}")
    print(f"Min height: {min_height}")
    print(f"Min width: {min_width}")
    print(f"Input shape: {input_shape}")

    return training_set, validation_set, input_shape

def main():
    ARGS, base_model = model_config()
    if base_model is None:
        print(f"{ARGS.arch} not in Keras applications!")
        print(f"Use '--help' for list of supported options.")
        return

    f_name = ARGS.data_csv.split("/")[-1].replace(".csv","")
    if ARGS.instance_type:
        log_name = f"./{f_name}_{ARGS.arch}_{ARGS.instance_type}_log.txt"
    else:
        log_name = f"./{f_name}_{ARGS.arch}_log.txt"

    LOG = open(log_name, "w")
    ini_time = datetime.now()
    LOG.write(f"Init time: {ini_time}\n\n")
    LOG.write(f"Tensorflow version: {tf.__version__}\n")

    # Pull the list of files
    train_df = pd.read_csv(ARGS.data_csv)
    images = [os.path.join(ARGS.data_dir, name) for name in train_df[ARGS.image_column].to_list()]
    labels = train_df[ARGS.label_column].to_list()

    if ARGS.classes is None:
        #classes = len(np.unique(labels))
        classes = max(labels) + 1
    else:
        classes = ARGS.classes

    # Split training/test sets
    # Returns ImageSet instances for each set and input shape
    training_set, validation_set, input_shape = split_and_resize(
        images=images,
        labels=labels,
        test_ratio=ARGS.test_ratio,
        input_shape=(ARGS.image_height, ARGS.image_width, ARGS.image_channels),
        auto_resize=ARGS.auto_resize,
        index_first=ARGS.index_first,
        log=LOG)

    # Create a mirrored strategy
    cdo_dict = {
        "all": tf.distribute.NcclAllReduce(),
        "hierarchical": tf.distribute.HierarchicalCopyAllReduce(),
        "one": tf.distribute.ReductionToOneDevice(reduce_to_device="/gpu:0")
    }
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=cdo_dict[ARGS.cross_dev_ops])
    LOG.write(f"Number of devices: {strategy.num_replicas_in_sync}\n")

    # Build the model
    classifier_activation = 'sigmoid'
    loss_type = 'sparse_categorical_crossentropy'
    lst_metrics = ['sparse_categorical_accuracy']
    lr_rate = 0.01

    with strategy.scope():
        #model = build_image_classifier(
        #    base_model=base_model,
        #    classes=classes,
        #    input_shape=input_shape,
        #    classifier_activation=classifier_activation,
        #    dropout=0.1)

        model = base_model(
            weights=None,
            classes=classes,
            input_shape=input_shape,
            classifier_activation=classifier_activation,
            include_top=True)

        opt = tf.keras.optimizers.Adam(learning_rate=lr_rate)

        model.compile(
            loss=loss_type,
            optimizer=opt,
            metrics=lst_metrics,
            run_eagerly=ARGS.run_eagerly)

    # Print Model Summary
    print(model.summary())

    # Determine batch size if auto-batch enabled
    # Auto-batch will not run if no GPU present
    if not ARGS.auto_batch or not tf.config.list_physical_devices('GPU'):
        batch_size = ARGS.batch_size
    else:
        batch_size = get_max_batch_size(model, unit="mebi", log=LOG)

    # Initialize settings for training
    train_steps = int(np.ceil(training_set.count / batch_size))
    val_steps = int(np.ceil(validation_set.count / batch_size))

    # Log steps
    LOG.write(f"train_steps: {train_steps}\n")
    LOG.write(f"val_steps: {val_steps}\n\n")

    # Create the data generators
    train_dataset = tf.data.Dataset.from_generator(
        training_set.generate_dataset,
        output_signature=(
            tf.TensorSpec(shape=training_set.input_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )).batch(batch_size, drop_remainder=False)

    val_dataset = tf.data.Dataset.from_generator(
        validation_set.generate_dataset,
        output_signature=(
            tf.TensorSpec(shape=validation_set.input_shape, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )).batch(batch_size, drop_remainder=False)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_dataset = train_dataset.with_options(options)
    val_dataset = val_dataset.with_options(options)

    # Train the model
    train_start_time = datetime.now()
    LOG.write(f"Training start time: {train_start_time}\n")
    LOG.write(f"Elapsed: {train_start_time - ini_time}\n")

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        './' + ARGS.output + '.h5',
        monitor='sparse_categorical_accuracy',
        verbose=1,
        save_best_only=True)
    epoch_time_callback = EpochTimeCallback(log=LOG)

    H = model.fit(
        x=train_dataset,
        steps_per_epoch=train_steps,
        validation_data=val_dataset,
        validation_steps=val_steps,
        epochs=ARGS.epochs,
        batch_size=batch_size,
        callbacks=[model_checkpoint, epoch_time_callback])

    # Save loss history
    loss_history = np.array(H.history['loss'])
    np.savetxt('./' + ARGS.output + '_loss.csv', loss_history, delimiter=",")

    end_time = datetime.now()
    LOG.write(f"\nEnd time: {end_time}\n")
    LOG.write(f"Training time: {end_time - train_start_time}\n")
    LOG.write(f"Total elapsed: {end_time - ini_time}")

    LOG.close()

if __name__ == '__main__':
    main()
