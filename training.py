import tensorflow as tf
import display
import process_functions as pf
import files_functions as ff
import tensorflow.keras.models as km
import tensorflow.keras.layers as layers
import tensorflow.keras.callbacks as call


tf.enable_eager_execution()  # to avoid using Session
CATEGORIES = ff.get_categories()  # we take all the categories at first
SIZE = 200  # size of images: SIZE * SIZE
LEARNING_RATE = 5e-8  # the model's learning rate
BATCH_SIZE = 1  # the batch size of data to train the model, 36 is good
MODEL_NAME = "dogs_6"  # the name that will be taken for the model's save
GRAY_SCALE = False  # to know if we use grayscale images  to feed the model
DOGS_6 = True  # to know if we take only 6 breeds of dogs to test the model or all breeds
STEPS_PER_EPOCH = 129  # steps per epoch, 100 is good for large dataset, else use less
EPOCHS = 6  # number of epochs to train the model

tensorboard = call.TensorBoard(log_dir='logs/{}'.format('dogs6_2'))


def train(learning_rate, gray_scale=False, epochs=5, batch_size=32, steps_per_epoch=100, model_name="dogs", dogs_6=False):
    """ Train the model with data contained in the dataset directory.

    Args:
        learning_rate: the learning rate used by the optimizer of the model
        gray_scale: boolean to know if we need to use grayscale image for inputs
        epochs: number of epochs to train the model, 5 by default
        batch_size: the batch size for data to train the model, 20 by default
        steps_per_epoch: number of steps per epoch, 100 by default
        model_name: the name used to save the model, dogs by default
        dogs_6: boolean to know if we need to use only 6 breeds to test the model
    Return:
        a trained model
    """

    label_training_ds, image_training_ds, label_validation_ds, image_validation_ds \
        = load_data(CATEGORIES, gray_scale, dogs_6=dogs_6)

    model = km.Sequential()
    model.add(layers.Conv2D(64, (4, 4), input_shape=(SIZE, SIZE, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    model.add(layers.Activation("relu"))

    model.add(layers.Conv2D(128, (4, 4)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    model.add(layers.Activation("relu"))

    model.add(layers.Conv2D(256, (4, 4)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(320, activation=tf.nn.relu))

    model.add(layers.Flatten())
    model.add(layers.Dense(5000, activation=tf.nn.relu))

    model.add(layers.Dense(2500, activation=tf.nn.relu))

    model.add(layers.Dense(1000, activation=tf.nn.relu))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(len(CATEGORIES), activation=tf.nn.softmax))

    model.compile(
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    training_ds = tf.data.Dataset.zip((image_training_ds, label_training_ds))
    validation_ds = tf.data.Dataset.zip((image_validation_ds, label_validation_ds))
    # model.fit(training_ds, batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch,
    #          validation_data=validation_ds, validation_steps=100)

    model.fit(training_ds, batch_size=batch_size, epochs=epochs,
              steps_per_epoch=steps_per_epoch, callbacks=[tensorboard])

    if dogs_6:
        validation_loss, validation_accuracy = model.evaluate(validation_ds, steps=41)
    else:
        validation_loss, validation_accuracy = model.evaluate(validation_ds, steps=300)

    print("Validation_loss: {}, Validation_accuracy: {}".format(validation_loss, validation_accuracy))

    ff.save_model(model, model_name)

    return model


def load_data(categories, gray_scale, dogs_6):
    """ Load all datas from the dataset

        Args:
            categories: all the dogs breeds
            gray_scale: boolean to know if we need to take grayscale images for inputs
            dogs_6: boolean to know if we need to take only 6 breeds of dogs to test the model
        Return:
            label_ds: tf.data.Dataset containing each label for image_ds (already shuffled)
            image_ds: tf.data.Dataset containing each image in the same order than labels (already shuffled)
    """

    if dogs_6:
        labels_training, images_path_training, labels_validation, images_path_validation, \
            number_elements_training, number_elements_validation = ff.get_files_for_some_categories(categories)
    else:
        labels_training, images_path_training, labels_validation, images_path_validation, \
            number_elements_training, number_elements_validation = ff.get_files(categories)

    # Essai d'appliquer une fonction à tout un numpy array
    # try:
    #     image_training_ds = pf.load_preprocess_image(images_path_training)
    #     image_validation_ds = pf.load_preprocess_image(images_path_validation)
    # except Exception as e:
    #     print(e)
    #     exit()

    # Essai en prenant un bach de tout puis get_next
    #path_image_training_ds = tf.data.Dataset.from_tensor_slices(images_path_training)
    #path_image_validation_ds = tf.data.Dataset.from_tensor_slices(images_path_validation)
    #image_training_ds = path_image_training_ds.map(pf.load_preprocess_image)
    #image_validation_ds = path_image_validation_ds.map(pf.load_preprocess_image)
    #image_training_ds = image_training_ds.batch(number_elements_training)
    #image_validation_ds = image_validation_ds.batch(number_elements_validation)
    #image_training_ds = image_training_ds.make_one_shot_iterator().get_next()
    #image_training_ds = image_training_ds.eval()
    #image_validation_ds = image_validation_ds.eval()
    #image_training_ds = image_training_ds.reshape(-1, 192, 192, 3)
    #image_validation_ds = image_validation_ds.reshape(-1, 192, 192, 3)

    path_image_training_ds = tf.data.Dataset.from_tensor_slices(images_path_training)
    label_training_ds = tf.data.Dataset.from_tensor_slices(labels_training)
    path_image_validation_ds = tf.data.Dataset.from_tensor_slices(images_path_validation)
    label_validation_ds = tf.data.Dataset.from_tensor_slices(labels_validation)

    image_training_ds = path_image_training_ds.map(
        lambda x: pf.load_preprocess_image(x, size=SIZE, gray_scale=gray_scale))
    image_validation_ds = path_image_validation_ds.map(
        lambda x: pf.load_preprocess_image(x, size=SIZE, gray_scale=gray_scale))

    # Essai foiré sur reshape -> bach mieux
    # image_training_ds = tf.reshape(image_training_ds, shape=[-1, 192, 192, 3])
    # image_validation_ds = tf.reshape(image_validation_ds, shape=[-1, 192, 192, 3])

    image_training_ds = image_training_ds.batch(1)
    image_validation_ds = image_validation_ds.batch(1)
    label_training_ds = label_training_ds.batch(1)
    label_validation_ds = label_validation_ds.batch(1)

    #image_training_ds = image_training_ds.map(lambda x: tf.cast(x, tf.int64))
    #image_validation_ds = image_validation_ds.map(lambda x: tf.cast(x, tf.int64))
    label_training_ds = label_training_ds.map(lambda x: tf.cast(x, tf.int64))
    label_validation_ds = label_validation_ds.map(lambda x: tf.cast(x, tf.int64))

    return label_training_ds, image_training_ds, label_validation_ds, image_validation_ds
    #return labels_training, image_training_ds, labels_validation, image_validation_ds


def predict_image_with_model(image_name, model_name, is_display=True):
    """ Give the label for an image contained in to_predict with a specific model.

     Args:
         image_name: name of the image the model need to predict
         model_name: the model which will be used to predict
         is_display: to know if the function display the image or not, True by default
     """

    model = ff.load_model(model_name)
    image = pf.load_preprocess_image(ff.get_predict_image_path(image_name))

    prediction = model.predict([image])
    prediction = tf.argmax(prediction)
    prediction = CATEGORIES(prediction)
    print(prediction)

    if is_display:
        display.display_image(image)


# labels_tr, image_tr, labels_val, image_val = load_data(CATEGORIES)
# train(LEARNING_RATE, steps_per_epoch=STEPS_PER_EPOCH, gray_scale=GRAY_SCALE, batch_size=BATCH_SIZE,
#      model_name=MODEL_NAME, dogs_6=DOGS_6, epochs=EPOCHS)

predict_image_with_model('blabla.jpg', 'dogs_6')
