""" COPDGene specific utilities """
import os
import glob
import random
import multiprocessing
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage import filters


def pull_random_nrrds(parent_dir, insp_exp, std_sharp, num_files):
    file_count = 0
    file_list = list()
    subject_list = glob.glob(os.path.join(parent_dir, '*/'))

    if num_files > len(subject_list):
        print(f'Number requested exceeds number of subjects. Setting to maximum: {len(subject_list)} files')
        num_files = len(subject_list)

    while (len(file_count) < num_files) and (len(subject_list) > 0):
        subject_index = random.randrange(len(subject_list))
        subject = subject_list.pop(subject_index)
        file_name = glob.glob(os.path.join(subject, '*/*' + insp_exp + std_sharp + '.nrrd'))
        if not file_name:
            continue

        file_list.append(file_name)

    print(f'Returned {len(file_list)} files')
    return file_list


def copdgene_process(file_path, model, save_individual_images=False, **kwargs):
    """
    This was paused in progress, so it is broken as of right now. The intention is to
    output image stacks of the predictions to stay within the file limits of SBG. Not
    sure if this will be necessary, but other things took precedence so this will wait.
    """
    file_name = file_path.split('/')[-1].replace('.nrrd', '')
    image_stack = sitk.ReadImage(file_path)
    image_array = sitk.GetArrayFromImage(image_stack)
    copd_array = np.reshape(image_array, (image_array.shape[0], image_array.shape[1], image_array.shape[2], 1))

    num_images = copd_array.shape[0]
    output_height = copd_array.shape[1]
    output_width = copd_array.shape[2]
    output_channels = 3
    copd_norm = np.ndarray(shape=(num_images, output_height, output_width, output_channels), dtype='float32')

    # Read the NRRD file and convert to NumPy array
    for ind in range(num_images):
        img = filters.median(copd_array[ind])
        img = (img - np.amin(img))/(np.amax(img) - np.amin(img))

        for channel in range(output_channels):
            copd_norm[ind,:,:,channel] = img[...,0]

    copd_predict_results = model.predict(copd_norm, batch_size=10, verbose=1)

    copd_predict_stack = sitk.GetImageFromArray(copd_predict_results)

    save_predictions(copd_predict_stack, file_name)


    # Just plotting stuff. Don't leave this in the batch, we need to output nrrd stacks
    plt.figure(figsize=(10,10))
    for i in range(16):
        ind = ((i+1)*30)-1
        plt.subplot(4,4,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(copd_norm[ind], cmap='gray')
        plt.imshow(copd_predict_results[ind], cmap=plt.get_cmap('jet'), alpha=0.4)
    plt.show()


def save_predictions(predict_img, output_path, filetype='png', save_individual_images=False):
    writer = sitk.ImageFileWriter()
    if save_individual_images:
        list(map(lambda i: write_slices(writer, predict_img, output_path, filetype, i), range(predict_img.GetDepth())))
    else:
        writer.SetFileName(os.path.join(output_path, '_predict.nrrd'))
        writer.Execute(predict_img)


def write_slices(writer, new_img, output_path, filetype, i):
    image_slice = new_img[:, :, i]

    # Write to the output directory and add the extension to force
    # writing in chosen format.
    writer.SetFileName(os.path.join(output_path, str(i) + '.' + filetype))
    writer.Execute(image_slice)



def nifti_to_flow_array(file_list, image_height, image_width):
    """
    Process a list of Nifti files and return two Rank 4 Numpy arrays
    for using the Keras ImageDataGenerator.flow() method
    """
    image_batch = []
    for file_zip in file_list:
        nifti_num = int(file_zip[0].replace('.nii','')[-1])
        image_batch.append(parallel_slices(file_zip, nifti_num, image_height, image_width))

    image_array, mask_array = create_flow_array(image_batch, image_height, image_width)

    return image_array, mask_array


def find_slices_with_masks(ind, nifti_num, img, mask, image_batch, image_height, image_width):
    # This returns images which have corresponding masks
    mask_slice = mask[:,:,ind]
    if max(mask_slice) > 0:
        img_slice = img[:,:,ind]
        img_slice = (img_slice - min(img_slice))/(max(img_slice) - min(img_slice))
        img_slice = sitk.GetArrayFromImage(img_slice)
        img_slice = np.reshape(img_slice, (1, image_height, image_width, 1))
        img_slice = img_slice.astype(dtype='float32')
        mask_slice = sitk.GetArrayFromImage(mask_slice)
        mask_slice = np.reshape(mask_slice, (1, image_height, image_width, 1))
        mask_slice = mask_slice.astype(dtype='float32')
        image_batch[ind] = (nifti_num, ind, img_slice, mask_slice)

    return image_batch


def create_flow_array(image_batch, image_height, image_width, num_channels=1):
    """
    Batch indices:
    0: Nifti file number
    1: Slice number
    2: Image array
    3: Mask array

    Create a Rank 4 numpy array each for images and masks
    (batch size, height, width, channels=1)
    """
    image_array = np.empty(shape=(1, image_height, image_width, num_channels), dtype='float32')
    mask_array = np.empty(shape=(1, image_height, image_width, num_channels), dtype='float32')
    for nifti in image_batch:
        for entry in nifti.values():
            image_array = np.append(image_array, entry[2], axis=0)
            mask_array = np.append(mask_array, entry[3], axis=0)
   
    image_array = np.delete(image_array, 0, 0)
    mask_array = np.delete(mask_array, 0, 0)
        
    return image_array, mask_array



def parallel_slices(file_zip, nifti_num, image_height, image_width):
    """
    Iterating through the slices of each Nifti file one at a time is very slow.
    This processes the slices in parallel and returns a list of dictionaries,
    where each dictionary contains the results from a single Nifti file.
    """
    manager = multiprocessing.Manager()
    image_batch = manager.dict()
    jobs = []

    image = sitk.ReadImage(file_zip[0])
    mask = sitk.ReadImage(file_zip[1])
    for i in range(image.GetSize()[2]):
        p = multiprocessing.Process(target=find_slices_with_masks, args=(i, nifti_num, image, mask, image_batch, image_height, image_width))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    return image_batch
