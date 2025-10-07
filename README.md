# GIDL - Generalized Image Deep Learning 

## Purpose
This repo houses the initial scripts for building a deep learning app on BioData Catalyst powered by Seven Bridges. These scripts will be used to test training scalability, among other issues.

## Main script
`train.py` trains a model for image classification.
### Input arguments:
| Arg | Description | Type | Default | Required |
| --- | ----------- | ---- | ------ | -------- |
| --data_dir | Path to directory containing images | string |  | YES |
| --data_csv | Path to CSV file pointing to images/labels | string |  | YES |
| --image_column | Column name for images | string |  | YES |
| --label_column | Column name for labels | string |  | YES |
| --arch | Model architecture | string |  | YES |
| --test_ratio | Percentage for testing data | float | 0.3 |   |
| --epochs | Number of training epochs | int | 15 |   |
| --classes | Number of classes. If not specified, classes will be inferred from labels | int | None |   |
| --batch_size | Training batch size | int | 8 |   |
| --output | Specify file name for output | string | 'model' |   |
| --auto_resize | Auto-resize to min height/width of image set | store_true | False |   |
| --auto_batch | Auto-detect max batch size. Selecting this will override any specified batch size | store_true | False |   |
| --index_first | Set images to depth as the first index (uncommon) | store_true | False |   |

## Supported architectures
Most built-in Keras applications are supported. For more information, see [https://keras.io/api/applications/](https://keras.io/api/applications/).
| Arg | Model |
| --- | ----- |
| densenet121 | DenseNet121 |
| densenet169 | DenseNet169 |
| densenet201 | DenseNet201 |
| efficientnetb0 | EfficientNetB0 |
| efficientnetb1 | EfficientNetB1 |
| efficientnetb2 | EfficientNetB2 |
| efficientnetb3 | EfficientNetB3 |
| efficientnetb4 | EfficientNetB4 |
| efficientnetb5 | EfficientNetB5 |
| efficientnetb6 | EfficientNetB6 |
| efficientnetb7 | EfficientNetB7 |
| inceptionresnetv2 | InceptionResNetV2 |
| inceptionv3 | InceptionV3 |
| mobilenet | MobileNet |
| mobilenetv2 | MobileNetV2 |
| nasnetlarge | NASNetLarge |
| nasnetmobile | NASNetMobile |
| resnet101 | ResNet101 |
| resnet101v2 | ResNet101V2 |
| resnet152 | ResNet152 |
| resnet152v2 | ResNet152V2 |
| resnet50 | ResNet50 |
| resnet50v2 | ResNet50V2 |
| vgg16 | VGG16 |
| vgg19 | VGG19 |
| xception | Xception |

## Batch sizing
Using the `auto_batch` feature will calculate the maximum batch size based on the memory allocated by TensorFlow when the model is loaded
if a GPU is detected. This will override any user defined `--batch_size`. If no GPU is detected, it will revert to using `--batch_size`,
which defaults to 8 if not defined.

## Debug
`get_sizes.py --data_dir /path/to/dir/ --data_csv /path/to/file.csv --image_column image_path_column_name` will create a CSV containing the image name, SimpleITK image shape, and Numpy array shape. It will also print this information to the console.
