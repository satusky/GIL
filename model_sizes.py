""" Train VGG on image data """
import argparse
import tensorflow as tf
from src.utility import get_max_batch_size

def main():
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
        "vgg16": tf.keras.applications.VGG16,
        "vgg19": tf.keras.applications.VGG19,
        "xception": tf.keras.applications.Xception
    }

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_height", help="Height of images (pixels)", required=True, type=int, default=512)
    parser.add_argument("--image_width", help="Width of images (pixels)", required=True, type=int, default=512)
    parser.add_argument("--image_channels", help="Number of channels", required=True, type=int, default=1)
    parser.add_argument("--classes", help="Number of classes", type=int, default=2)
    parser.add_argument("--cross_dev_ops", help="Cross device operation to use for multi-GPU reduction. 'all' = NcclAllReduce, 'hierarchical' = HierarchicalCopyAllReduce, 'one' = ReductionToOneDevice", type=str, choices=["all", "hierarchical", "one"], default="hierarchical")
    ARGS = parser.parse_args()

    LOG = open("model_sizes.txt", "w")
    LOG.write(f"Tensorflow version: {tf.__version__}\n")

    # Create a mirrored strategy
    cdo_dict = {
        "all": tf.distribute.NcclAllReduce(),
        "hierarchical": tf.distribute.HierarchicalCopyAllReduce(),
        "one": tf.distribute.ReductionToOneDevice(reduce_to_device="/gpu:0")
    }
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=cdo_dict[ARGS.cross_dev_ops])

    # Build the model
    classifier_activation = 'sigmoid'
    loss_type = 'sparse_categorical_crossentropy'
    lst_metrics = ['sparse_categorical_accuracy']
    lr_rate = 0.01
    input_shape = (ARGS.image_height, ARGS.image_width, ARGS.image_channels)

    for arch, base_model in model_dict.items():
        LOG.write(f"Model: {arch}\n")
        with strategy.scope():
            #model = build_image_classifier(
            #    base_model=base_model,
            #    classes=ARGS.classes,
            #    input_shape=input_shape,
            #    classifier_activation=classifier_activation,
            #    dropout=0.1)

            model = base_model(
                weights=None,
                classes=ARGS.classes,
                input_shape=input_shape,
                classifier_activation=classifier_activation,
                include_top=True)

            opt = tf.keras.optimizers.Adam(learning_rate=lr_rate)

            model.compile(
                loss=loss_type,
                optimizer=opt,
                metrics=lst_metrics)

        # Determine batch size if auto-batch enabled
        # Auto-batch will not run if no GPU present
        _ = get_max_batch_size(model, unit="mebi", log=LOG)

    LOG.close()

if __name__ == '__main__':
    main()
