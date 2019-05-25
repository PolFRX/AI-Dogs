import os
import numpy as np
import math


def get_files():
    """ To get all files present in the directory 'dataset'.

    Return:
        labels_training: numpy array which contains labels of training images
        image_path_training: numpy array which contains paths of training images
        labels_validation: numpy array which contains labels of validation images
        image_path_validation: numpy array which contains paths of validation image

     """
    # Take the path to directory "dataset"
    path = os.getcwd()
    path = path + "\\dataset"
    dataset_dirs = os.listdir(path)
    # Declare the two list which will contain labels and path of the images
    labels = list()
    image_path = list()

    # Browse the "dataset" directory to get all paths for images and labels which are the directories names
    for dirs in dataset_dirs:
        dir_path = path + "\\" + dirs
        for path2, dir, files in os.walk(dir_path):
            for files_name in files:
                labels.append(dirs)
                image_path.append(dir_path+"\\"+files_name)

    labels = np.array(labels)
    image_path = np.array(image_path)

    # Shuffle the two arrays with the same permutation
    permutation = np.random.permutation(len(labels))
    labels = labels[permutation]
    image_path = image_path[permutation]

    # Separate training and validation data with 80% of training data and 20% of validation.
    length_training_data = math.floor(len(labels) * 0.8)
    labels_training = labels[:length_training_data]
    labels_validation = labels[length_training_data - len(labels):]
    image_path_training = image_path[:length_training_data]
    image_path_validation = image_path[length_training_data - len(labels):]

    return labels_training, image_path_training, labels_validation, image_path_validation
