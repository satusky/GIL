""" Return image sizes """
import argparse
from tqdm import tqdm
import numpy as np
import pandas as pd
import SimpleITK as sitk


def print_images_with_sizes(file_list):
    file_size_df = pd.DataFrame(columns=['image', 'image_size', 'array_size'])
    reader = sitk.ImageFileReader()

    with tqdm(total=len(file_list)) as pbar:
        for i,file in enumerate(file_list):
            reader.SetFileName(file)
            reader.LoadPrivateTagsOn()
            reader.ReadImageInformation()
            file_size_df.loc[i,'image'] = file
            file_size_df.loc[i,'image_size'] = reader.GetSize()
            image = sitk.ReadImage(file)
            image_array = sitk.GetArrayFromImage(image)
            file_size_df.loc[i,'array_size'] = image_array.shape
            pbar.update(1)

    return file_size_df


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv', required=True, metavar='CSV FILE', help="CSV file pointing to images" )
    parser.add_argument('--image_column', required=True, help='Column name for images')

    args = parser.parse_args()

    # Point to images
    image_list_file = args.data_csv
    image_column = args.image_column

    # Pull the list of files
    image_df = pd.read_csv(image_list_file)
    images =  image_df[image_column].to_list()
    size_df = print_images_with_sizes(images)
    size_df.to_csv('image_sizes.csv', index=False)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(size_df)
