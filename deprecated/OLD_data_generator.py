""" COPDGene training data generator """
import numpy as np
import SimpleITK as sitk
import tensorflow as tf

class ImageSet:
    """An image stack dataset for batch generation in TensorFlow"""
    def __init__(self, images, labels, classes, index_first=False):
        """
        Constructs an ImageSet from a list of stack files.

        Attributes:
            images : list
                List of image file paths
            labels : list
                List of labels (one per stack)
            index_first : bool
                Defaults to False. Set True if the depth index
                is first in the image size (not common)
            count : int
                Number of images in all stacks
            min_height : int
                Minimum image height in set
            min_width : int
                Minimum image width in set
        """
        self.images = images
        self.labels = labels
        self.classes = classes
        self.index_first = index_first
        self.count = None
        self.min_height = None
        self.min_width = None
        self.get_sizes()

    def get_sizes(self):
        file_size_list = []
        reader = sitk.ImageFileReader()
        width_index = 1 if self.index_first else 0
        height_index = 2 if self.index_first else 1
        depth_index = 0 if self.index_first else -1
        min_height = None
        min_width = None

        for image in self.images:
            reader.SetFileName(image)
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

        self.count = sum(file_size_list)
        self.min_height = min_height
        self.min_width = min_width

    def generate_batches(self, batch_size, input_shape):
        """
        This is a custom data generator for SimpleITK image stacks.
        'Index first' is relative to the SimpleITK image; if it is False, the first
        position will be the depth index of the NumPy array because of the shape convention difference.
        """
        height = input_shape[0]
        width = input_shape[1]
        depth_index = -1 if self.index_first else 0

        # Load first stack
        file_index = 0
        slice_num = 0
        img_array, img_label, file_index = self.process_next_stack(file_index, height, width)

        # Loop indefinitely
        while True:
            # Initialize image batch
            batch_array = []
            labels = []

            # Populate array until we hit the batch size
            while len(batch_array) < batch_size:
                if slice_num < img_array.shape[depth_index]-1:
                    if self.index_first:
                        batch_array.append(img_array[:,:,slice_num])
                    else:
                        batch_array.append(img_array[slice_num,:,:])
                    labels.append(img_label)
                    slice_num += 1
                else:
                    img_array, img_label, file_index = self.process_next_stack(file_index, height, width)
                    slice_num = 0

            # Set correct formats
            batch_array = np.array(batch_array)
            #print(f'batch_array shape BEFORE: {batch_array.shape}')
            batch_array = np.reshape(batch_array, (batch_array.shape[0], batch_array.shape[1], batch_array.shape[2], 1))
            #print(f'batch_array shape AFTER: {batch_array.shape}')
            labels = tf.keras.utils.to_categorical(labels, num_classes=self.classes)

            # Yield to the calling function
            yield (batch_array, labels)

    def generate_prediction_batches(self, batch_size, input_shape, small_set=32):
        height = input_shape[0]
        width = input_shape[1]
        depth_index = 0 if self.index_first else -1

        # Load first stack
        file_index = 0
        slice_num = 0
        slice_all = 0
        counter = 0

        img_array, img_label, file_index = self.process_next_stack(file_index, height, width)

        img_array = img_array[:,:,:small_set]

        # Loop indefinitely
        while slice_all < img_array.shape[-1]-1:
            # Initialize image batch
            batch_array = []
            labels = []

            # Populate array until we hit the batch size
            while len(batch_array) < min(batch_size, small_set):
                if slice_num < img_array.shape[-1]-1:
                    batch_array.append(img_array[:,:,slice_num])
                    labels.append(img_label)
                    slice_num += 1
                else:
                    if len(self.images) > 1:
                        img_array, img_label, file_index = self.process_next_stack(file_index, height, width)
                    slice_all += slice_num
                    counter += 1
                    slice_num = 0

            # Set correct formats
            batch_array = np.array(batch_array)
            batch_array = np.reshape(batch_array, (batch_array.shape[0], batch_array.shape[1], batch_array.shape[2], 1))

            labels = tf.keras.utils.to_categorical(labels)

            # Yield to the calling function
            yield (batch_array, labels)

    def process_next_stack(self, file_index, height, width):
        # Load stack file
        img = sitk.ReadImage(self.images[file_index])
        # Convert to img_array
        img_array = sitk.GetArrayFromImage(img)
        # Convert RGB to grayscale
        if len(img_array.shape) == 4:
            img = self._rgb_to_gray(img)
            img_array = sitk.GetArrayFromImage(img)
        # Expand dimensions to create (depth, width, height, channel)
        img_array = np.expand_dims(img_array, 3)
        # Resize image array to target size
        img_array = tf.image.resize(img_array, [height,width]).numpy()
        # Normalize 0-1
        img_array = (img_array - np.min(img_array))/(np.max(img_array) - np.min(img_array))
        # Pull label
        img_label = self.labels[file_index]

        # If not at the end of the file list, queue the next stack; otherwise loop to the beginning
        if file_index < len(self.images)-1:
            file_index += 1
        else:
            file_index = 0

        return img_array, img_label, file_index

    def _rgb_to_gray(self, image):
        # Convert sRGB image to gray scale and rescale results to [0,255]    
        channels = [sitk.VectorIndexSelectionCast(image, i, sitk.sitkFloat32) for i in range(image.GetNumberOfComponentsPerPixel())]
        #linear mapping
        I = 1/255.0*(0.2989*channels[0] + 0.5870*channels[1] + 0.1140*channels[2])

        return sitk.Cast(sitk.RescaleIntensity(I), sitk.sitkUInt8)
