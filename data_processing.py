import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from glob import glob
from pickle import dump


def is_pneumonic(path):
    '''Returns whether the patient is pneumonic or not
    '''
    if 'PNEUMONIA' in path:
        return np.uint8(1)

    else:
        return np.uint8(0)

##############################################################################


def scale_pixel_values(input_array):
    '''Min max scale each pixel value. Range becomes [0, 1]
    '''
    # grayscale range:
    # black => 0
    # white => 255
    #########################
    return input_array / 255

##############################################################################


def process_image(path):
    '''Convert the input image (.jpeg) into a scaled numpy array
    '''
    img = load_img(
        path=path,
        # reformat images to so that they all have the same size
        # values represent pixels
        target_size=(150, 250),
        color_mode='grayscale'
    )

    # convert each image into a 2D array, grayscale color coded (0 to 255)
    features_array = np.array(img)

    # min max scale the array (image)
    features_array = scale_pixel_values(features_array)

    # return the feature variables
    return features_array

#############################################################################


def get_paths(folder):
    '''Get the relative path to each file inside a folder
    Returns a list of relative paths
    '''
    pattern = 'data/raw_data/' + folder + '/*/*.jpeg'

    return glob(pattern)

#############################################################################


def get_sampled_paths(img_paths, threshold=1/3, max_images=20):
    '''Return a subset (1D np array) of the image paths to speed up computations
    '''
    # define a threshold for subsetting or not the list of image paths
    threshold = int(len(img_paths)*threshold)
    # condition to decide on subsetting or not
    sample_size = threshold if threshold > max_images else len(img_paths)
    # ensuring reproducible results
    np.random.seed(0)
    # return a subset of the images to speed up computations
    return np.random.choice(np.array(img_paths),
                            size=sample_size,
                            replace=False)

#############################################################################


def process_folder(folder):
    '''Process all images (.jpeg) nested inside a folder
    Returns a tuple containing a 4D features array and a 1D targets array
    '''

    save_features = []
    save_targets = []
    #########################
    img_paths = get_sampled_paths(img_paths=get_paths(folder))
    #########################
    for path in img_paths:

        features, target = process_image(path), is_pneumonic(path)

        save_features.append(features)
        save_targets.append(target)

    # return the features and the target variable
    return (
        # concatenate 2D arrays into 3D arrays
        # then set the 3D array into a 4D array (adding an axis) which is needed while using a CNN Model
        # => each image [28, 28, 1] represents a 28X28 image with one color (grayscale) channel (e.g. for RGB this value would be 3)
        np.stack(save_features)[..., np.newaxis],
        np.array(save_targets)  # return a 1D array
    )

#############################################################################


def process_all_folders(*args):
    '''Process images from all input folders
    '''
    return {folder: process_folder(folder) for folder in args}

#############################################################################


# saving the processed data
with open('data/processed_data/inputs.pickle', 'wb') as file:
    dump(process_all_folders('train', 'test', 'val'), file)
