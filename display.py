import matplotlib.pyplot as plt

def display_image(img):
    """ Show an image

     Args:
         img: a preprocessed image
     """

    fig = plt.figure()
    fig.suptitle("Photo cute")

    original_plt = fig.add_subplot(1, 2, 1)
    original_plt.set_title('original image')
    original_plt.imshow(img)

    plt.show()


def display_all_images_with_labels(labels, images):
    """ Show all images with their labels

     Args:
         labels: a TensorSliceDataset containing all labels
         images: a tfTensor containing all preprocessed images
     """
    iter_labels = labels.make_one_shot_iterator()
    plt.figure(figsize=(8, 8))
    for n, img in enumerate(images):
        plt.subplot(2, 2, n + 1)
        plt.imshow(img)
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
        plt.xlabel(iter_labels.get_next().numpy().decode('utf-8'))

    plt.show()

