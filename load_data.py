import numpy as np
import math
import os
import pickle
import sys

data_path = "/Users/Ben/data/CIFAR-10/"

def one_hot_encoded(class_numbers, num_classes=None):
    num_classes = np.max(class_numbers) + 1
    return np.eye(num_classes, dtype=float)[class_numbers]

def _get_file_path(filename=""):
    return os.path.join(data_path, "cifar-10-batches-py/", filename)

def _unpickle(filename):
    file_path = _get_file_path(filename)
    print("Loading data: " + file_path)

    with open(file_path, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')

    return data

def _convert_images(raw):
    raw_float = np.array(raw, dtype=float) / 255.0

    images = raw_float.reshape([-1, 3, 32, 32])

    images = images.transpose([0, 2, 3, 1])

    return images

def _load_data(filename):
    data = _unpickle(filename)

    raw_images = data[b'data']

    cls = np.array(data[b'labels'])

    images = _convert_images(raw_images)

    return images, cls

#def load_class_names():
 #   raw = _unpickle(filename="batches.meta")[b'label_names']

  #  names = [x.decode('utf-8') for x in raw]

 #   return names

def load_training_data():
    images = np.zeros(shape=[50000, 32, 32, 3], dtype=float)
    cls = np.zeros(shape=[50000], dtype=int)

    begin = 0

    for i in range(5):
        images_batch, cls_batch = _load_data(filename="data_batch_" + str(i + 1))

        num_images = len(images_batch)

        end = begin + num_images

        images[begin:end, :] = images_batch

        cls[begin:end] = cls_batch

        begin = end

    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=10)

def load_test_data():
    images, cls = _load_data(filename="test_batch")

    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=10)

xtrain = np.empty((50000, 32, 32, 3))

ytrain = np.empty(50000)

xtest = np.empty((10000, 32, 32, 3))

ytest = np.empty(10000)

xtrain, ytrain, _ = load_training_data()

xtest, ytest, _ = load_test_data()

np.save('/Users/Ben/data/CIFAR-10/xtrain', xtrain, allow_pickle=False)
np.save('/Users/Ben/data/CIFAR-10/ytrain', ytrain, allow_pickle=False)
np.save('/Users/Ben/data/CIFAR-10/xtest', xtest, allow_pickle=False)
np.save('/Users/Ben/data/CIFAR-10/ytest', ytest, allow_pickle=False)