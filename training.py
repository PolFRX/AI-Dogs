import tensorflow as tf
import display, process_functions as pf, files_functions as ff
import tensorflow.keras.models as km
import tensorflow.keras.layers as layers

tf.enable_eager_execution()
CATEGORIES = ff.get_categories()


def train(learning_rate, epochs=5, batch_size=1, steps_per_epoch=100, model_name="dogs"):
    """ Train the model with data contained in the dataset directory.

    Args:
        learning_rate: the learning rate used by the optimizer of the model
        epochs: number of epochs to train the model, 5 by default
        batch_size: the batch size for data to train the model, 20 by default
        steps_per_epoch: number of steps per epoch, 100 by default
        model_name: the name used to save the model, dogs by default
    Return:
        a model trained
    """

    label_training_ds, image_training_ds, label_validation_ds, image_validation_ds = load_data(CATEGORIES)

    model = km.Sequential()
    model.add(layers.Conv2D(64, (3, 3), input_shape=(192, 192, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(len(CATEGORIES), activation=tf.nn.softmax))

    model.compile(
        optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    training_ds = tf.data.Dataset.zip((image_training_ds, label_training_ds))
    model.fit(training_ds, batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch)
    # model.fit(image_training_ds, label_training_ds, batch_size=batch_size, epochs=epochs,
    #          steps_per_epoch=steps_per_epoch)
    validation_loss, validation_accuracy = model.evaluate(image_validation_ds, label_validation_ds)
    print("Validation_loss: {}, Validation_accuracy: {}".format(validation_loss, validation_accuracy))

    ff.save_model(model, model_name)

    return model


def load_data(categories):
    """ Load all datas from the dataset

        Return:
            label_ds: tf.data.Dataset containing each label for image_ds (already shuffled)
            image_ds: tf.data.Dataset containing each image in the same order than labels (already shuffled)
    """

    labels_training, images_path_training, labels_validation, images_path_validation = ff.get_files(categories)

    path_image_training_ds = tf.data.Dataset.from_tensor_slices(images_path_training)
    label_training_ds = tf.data.Dataset.from_tensor_slices(labels_training)
    path_image_validation_ds = tf.data.Dataset.from_tensor_slices(images_path_validation)
    label_validation_ds = tf.data.Dataset.from_tensor_slices(labels_validation)

    image_training_ds = path_image_training_ds.map(pf.load_preprocess_image)
    image_validation_ds = path_image_validation_ds.map(pf.load_preprocess_image)

    return label_training_ds, image_training_ds, label_validation_ds, image_validation_ds


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
    prediction = CATEGORIES.index(prediction)
    print(prediction)

    if is_display:
        display.display_image(image)

