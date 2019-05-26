import tensorflow as tf


def load_preprocess_image(path):
    """ Load an image and then call process_image to return a dtype """
    img = tf.read_file(path)
    return preprocess_image(img)


def preprocess_image(raw_img):
    """ Decode an image, resize it and then put each value between 0 and 1 """
    img = tf.cond(
        tf.image.is_jpeg(raw_img),
        lambda: tf.image.decode_jpeg(raw_img, channels=3),
        lambda: tf.image.decode_png(raw_img, channels=3)
    )
    img = tf.image.resize_images(img, [192, 192])
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = img/255.0
    return img
