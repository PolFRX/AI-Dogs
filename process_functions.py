import tensorflow as tf
import pandas as pd
import numpy as np

def load_preprocess_image(path):
    """ Load an image and then call process_image to return a dtype """
    img = tf.read_file(path)
    return preprocess_image(img)

def preprocess_image(raw_img):
    """ Decode an image, resize it and then put each value between 0 and 1 """
    img = tf.image.decode_image(raw_img)
    img = tf.image.resize_images(img, [192, 192])
    img = img/255.0
    return img