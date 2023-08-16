from pix2pix_model import define_discriminator, define_generator, define_gan, train
from numpy import vstack
from matplotlib import pyplot
from os import listdir
from keras.preprocessing.image import img_to_array, load_img
from os.path import join
import numpy as np


def load_images(path, size=(256, 256)):
    src_list, tar_list = list(), list()

    src_path = join(path, 'features/')
    tar_path = join(path, 'labels/')

    src_filenames = listdir(src_path)
    tar_filenames = listdir(tar_path)

    for src_filename, tar_filename in zip(src_filenames, tar_filenames):
        src_pixels = load_img(join(src_path, src_filename), target_size=size)
        tar_pixels = load_img(join(tar_path, tar_filename), target_size=size)

        src_pixels = img_to_array(src_pixels)
        tar_pixels = img_to_array(tar_pixels)

        src_list.append(src_pixels)
        tar_list.append(tar_pixels)

    return [np.asarray(src_list), np.asarray(tar_list)]


# dataset path
path = 'dataset_3/'
# load dataset
[src_images, tar_images] = load_images(path)
print('Loaded: ', src_images.shape, tar_images.shape)

# save src_images and tar_images to file
np.save('src_images.npy', src_images)
np.save('tar_images.npy', tar_images)

# load src_images and tar_images from file
src_images = np.load('src_images.npy')
tar_images = np.load('tar_images.npy')

n_samples = 5
for i in range(n_samples):
    pyplot.subplot(2, n_samples, 1 + i)
    pyplot.axis('off')
    pyplot.imshow(src_images[i].astype('uint8'))
# plot target image
for i in range(n_samples):
    pyplot.subplot(2, n_samples, 1 + n_samples + i)
    pyplot.axis('off')
    pyplot.imshow(tar_images[i].astype('uint8'))
pyplot.show()

