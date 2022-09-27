""" MAIN """ 
import argparse
import os
import tensorflow as tf
import numpy as np
import pandas as pd

from vgg16 import VGG_16
from copdgene_data_generator import *

from utility import *

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# from tf.keras.callbacks import ModelCheckpoint
print('Tensorflow version: ' + tf.__version__)

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_directory', metavar='FOLDER',
                        required=True, help="Directory containing images.")
    parser.add_argument(
        '--insp_exp', help='Specify INSP(iration) or EXP(iration). Default is both', default='')
    parser.add_argument(
        '--std_sharp', help='Specify STD or SHARP images. Default is both', default='')
    parser.add_argument(
        '--num_files', help='Number of files to include in testing. Default is 100', type=int, default=100)
    parser.add_argument('--trained_model',
                        help='Trained Model .h5 file', required=True)
    parser.add_argument(
        '--batch_size', help='Training batch size. Default is 8', type=int, default=8)
    parser.add_argument('--num_classes', help='Number of classes', type=int, default=1)
    parser.add_argument('--name_analysis', help="Name your analysis test outputs", default="Test-Analysis_")

    args = parser.parse_args()

    insp_exp = args.insp_exp
    std_sharp = args.std_sharp
    num_files = args.num_files
    batch_size = args.batch_size
    trained_model = args.trained_model
    num_classes = args.num_classes
    name_analysis = args.name_analysis

    class_threshold = 0.5

    parameters_dict = {
        "name_analysis": name_analysis
    }

    # Point to project files folder
    parent_dir = args.data_directory

    # Pull the list of files
    predict_images, predict_labels = pull_random_nrrds(parent_dir, num_files=num_files)

    # FOR DEBUG REMOVE IT
    eprint(f"Train Shape: {len(predict_images)}")
    eprint(f"Train Label Len: {len(predict_labels)}")

    # Get total number of images in each set
    predict_image_sizes, predict_image_count = get_image_set_size(predict_images)

    # FOR DEBUG REMOVE IT
    eprint(f"train_image_sizes: {predict_image_sizes}")
    eprint(f"train_image_count: {predict_image_count}")

    # Create a mirrored strategy
    strategy = tf.distribute.MirroredStrategy()
    eprint(f'Number of devices: {strategy.num_replicas_in_sync}')

    # # Create the data generators
    predict_gen = batch_generator_predict(predict_images, predict_labels, batch_size=32, small_set=train_image_count)
    # img_array, img_label, file_index = process_next_nrrd(predict_images, predict_labels, 0)


    with strategy.scope():
        eprint('Loading model...')
        model = tf.keras.models.load_model(trained_model)
    
        eprint('Calculating metrics...')
        preds = model.predict_generator(predict_gen)

        # probs_preds = np.argmax(preds, axis=1)
        results = [1 if r > class_threshold else 0 for i,r in enumerate(preds) ]

        eprint('[DEBUG] Print Prediction Probabilies')
        eprint(results)

        test_input = [0]*(train_image_count-32) + [1]*32

        eprint('[PROGRESS] Calculating metrics...')

        eprint('[1] Calculating plot_auc...')
        plot_auc(test_input, preds, parameters_dict)

        eprint('[2] Calculating plot_predictions...')
        plot_predictions(test_input, results, parameters_dict)

        eprint('[3] Calculating calculate_metrics...')
        d = {}
        d['Test accuracy'], d['Test AUC'], d['Test AUPRC'], d['Test precision'], d['Test recall'], d['Test F-score'], d['Test MCC'] = calculate_metrics(test_input, preds, results)

        eprint('[4] Calculating plot_confusion_matrix...')
        cm_val = confusion_matrix(test_input, results)
        d['Test specificity'], d['Test fall-out'] = plot_confusion_matrix(cm_val, [0, 1], parameters_dict, "Confusion Matrix")

        eprint('[5] Calculating plot_confusion_matrix...')
        conf_col = ':'.join(parameters_dict.keys())
        columns = [conf_col] + list(d.keys())
        df_out = pd.DataFrame(columns=columns)
        df_out.loc[0] = [':'.join(map(str, parameters_dict.values()))] + list(d.values())
        df_out.to_csv(name_analysis + '.csv', index=False)
