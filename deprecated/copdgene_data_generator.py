""" COPDGene training data generator """
import os
import random
import glob
import numpy as np
import SimpleITK as sitk
import tensorflow as tf

def batch_generator(file_list, label_list, batch_size, input_shape, index_first=False):
    """
    This is a custom data generator for SimpleITK image stacks.
    'Index first' is relative to the SimpleITK image; if it is False, the first
    position will be the depth index of the NumPy array because of the shape convention
    difference.
    """
    height = input_shape[0]
    width = input_shape[1]
    depth_index = -1 if index_first else 0

    # Load first stack
    file_index = 0
    slice_num = 0
    img_array, img_label, file_index = process_next_stack(file_list, label_list, file_index, height, width)

    # Loop indefinitely
    while True:
        # Initialize image batch
        batch_array = []
        labels = []

        # Populate array until we hit the batch size
        while len(batch_array) < batch_size:
            if slice_num < img_array.shape[depth_index]-1:
                if index_first:
                    batch_array.append(img_array[:,:,slice_num])
                else:
                    batch_array.append(img_array[slice_num,:,:])
                labels.append(img_label)
                slice_num += 1
            else:
                img_array, img_label, file_index = process_next_stack(file_list, label_list, file_index, height, width)
                slice_num = 0

        # Set correct formats
        batch_array = np.array(batch_array)
        #print(f'batch_array shape BEFORE: {batch_array.shape}')
        batch_array = np.reshape(batch_array, (batch_array.shape[0], batch_array.shape[1], batch_array.shape[2], 1))
        #print(f'batch_array shape AFTER: {batch_array.shape}')
        labels = tf.keras.utils.to_categorical(labels)

        # Yield to the calling function
        yield (batch_array, labels)

def batch_generator_predict(file_list, label_list, batch_size, input_shape, index_first=False, small_set=32):
    height = input_shape[0]
    width = input_shape[1]
    depth_index = 0 if index_first else -1

    # Load first stack
    file_index = 0
    slice_num = 0
    slice_all = 0
    counter = 0

    img_array, img_label, file_index = process_next_stack(file_list, label_list, file_index, height, width)

    img_array = img_array[:,:,:small_set]

    # Loop indefinitely
    while slice_all < img_array.shape[-1]-1:
        # Initialize image batch
        batch_array = []
        labels = []

        # Populate array until we hit the batch size
        while len(batch_array) < min(batch_size, small_set):
            if slice_num < img_array.shape[-1]-1:#img_array.shape[0]-1:
                batch_array.append(img_array[:,:,slice_num])
                labels.append(img_label)
                slice_num += 1
            else:
                if len(file_list) > 1:
                    img_array, img_label, file_index = process_next_stack(file_list, label_list, file_index, height, width)
                slice_all += slice_num
                counter += 1
                slice_num = 0

        # Set correct formats
        batch_array = np.array(batch_array)

        print(f'batch_array shape BEFORE: {batch_array.shape} Counter: {counter}')
        batch_array = np.reshape(batch_array, (batch_array.shape[0], batch_array.shape[1], batch_array.shape[2], 1))

        print(f'batch_array shape AFTER: {batch_array.shape}')
        labels = tf.keras.utils.to_categorical(labels)

        # Yield to the calling function
        yield (batch_array, labels)

def process_next_stack(file_list, label_list, file_index, height, width):
    # Load stack file
    img = sitk.ReadImage(file_list[file_index])
    # Convert to img_array
    img_array = sitk.GetArrayFromImage(img)
    # Convert RGB to grayscale
    if len(img_array.shape) == 4:
        img = rgb2gray(img)
        img_array = sitk.GetArrayFromImage(img)
    # Expand dimensions to create (depth, width, height, channel)
    img_array = np.expand_dims(img_array, 3)
    # Resize image array to target size
    img_array = tf.image.resize(img_array, [height,width]).numpy()
    # Normalize 0-1
    img_array = (img_array - np.min(img_array))/(np.max(img_array) - np.min(img_array))
    # Pull label
    img_label = label_list[file_index]

    # If not at the end of the file list, queue the next stack; otherwise loop to the beginning
    if file_index < len(file_list)-1:
        file_index += 1
    else:
        file_index = 0

    return img_array, img_label, file_index

def rgb2gray(image):
    # Convert sRGB image to gray scale and rescale results to [0,255]    
    channels = [sitk.VectorIndexSelectionCast(image,i, sitk.sitkFloat32) for i in range(image.GetNumberOfComponentsPerPixel())]
    #linear mapping
    I = 1/255.0*(0.2989*channels[0] + 0.5870*channels[1] + 0.1140*channels[2])

    return sitk.Cast(sitk.RescaleIntensity(I), sitk.sitkUInt8)


def pull_random_nrrds(parent_dir, insp_exp='', std_sharp='', num_files=100):
    file_list = []
    file_labels = []

    subject_list = glob.glob(os.path.join(parent_dir, '*/'))
    ##### TODO: Replace the subject labels with some thing useful from PIC-SURE
    ##### For now just assign them a random 0 or 1 for a binary classifier
    subject_label_list = [random.randint(0, 1) for subject in subject_list]

    while (len(file_list) < num_files) and (len(subject_list) > 0):
        file_index = random.randrange(len(subject_list))
        subject = subject_list.pop(file_index)
        subject_label = subject_label_list.pop(file_index)
        file_name = glob.glob(os.path.join(subject, '*' + insp_exp + std_sharp + '.nrrd')) # EDITED
        labels = [subject_label for file in file_name]
        if not file_name:
            continue

        for file in file_name:
            file_list.append(file)
        for label in labels:
            file_labels.append(label)

    print(f'Returned {len(file_list)} files')
    return file_list, file_labels


def get_image_set_size(file_list, index_first=False):
    file_size_list = []
    reader = sitk.ImageFileReader()
    width_index = 1 if index_first else 0
    height_index = 2 if index_first else 1
    depth_index = 0 if index_first else -1
    min_height = None
    min_width = None

    for file in file_list:
        reader.SetFileName(file)
        reader.LoadPrivateTagsOn()
        reader.ReadImageInformation()
        img_shape = reader.GetSize()

        # Get min height/width
        if min_height is None:
            min_height = img_shape[height_index]
        elif img_shape[height_index] < min_height:
            min_height = img_shape[height_index]

        if min_width is None:
            min_width = img_shape[width_index]
        elif img_shape[width_index] < min_width:
            min_width = img_shape[width_index]

        file_size_list.append(img_shape[depth_index]) # (554, 512, 512)

    num_images = sum(file_size_list)

    return file_size_list, num_images, min_height, min_width
