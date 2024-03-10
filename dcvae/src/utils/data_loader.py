from src.libraries import *
import torchvision
import tempfile
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import tensorflow_datasets as tfds
import cv2

def data_generator(X, y):
    for image, label in zip(X, y):
        yield {'x': image, 'y': label}

def preprocess_images(images, dims=(-1, 28, 28, 1)):
    images = images.reshape(dims) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')

def preprocess_images_cifar10(images, dims=(-1, 32, 32, 1)):
    images = images.reshape(dims) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')

def get_dataset(task, train_size, test_size, batch_size):
    if task.lower() == 'mnist':
        num_classes = 10
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

        train_labels = tf.one_hot(train_labels, num_classes)
        test_labels = tf.one_hot(test_labels, num_classes)
        train_images = preprocess_images(train_images)
        test_images = preprocess_images(test_images)

    elif task.lower() == 'fmnist':
        num_classes = 10
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

        train_labels = tf.one_hot(train_labels, num_classes)
        test_labels = tf.one_hot(test_labels, num_classes)
        train_images = preprocess_images(train_images)
        test_images = preprocess_images(test_images)

    elif task.lower() == 'emnist':
        num_classes = 62
        (train_images, train_labels), (test_images, test_labels) = tfds.as_numpy(tfds.load('emnist',
                         split = ['train[:50%]', 'test[:50%]'],
                         batch_size=-1,
                         as_supervised=True))

        train_labels = tf.one_hot(train_labels, num_classes)
        test_labels = tf.one_hot(test_labels, num_classes)
        train_images = preprocess_images(train_images)
        #train_images = train_images.transpose().reshape(28,28)
        test_images = preprocess_images(test_images)
        #test_images = test_images.transpose().reshape(28,28)

    elif task.lower() == 'kmnist':
        num_classes = 10
        (train_images, train_labels), (test_images, test_labels) = tfds.as_numpy(tfds.load('kmnist',
                         split = ['train', 'test'],
                         batch_size=-1,
                         as_supervised=True))

        train_labels = tf.one_hot(train_labels, num_classes)
        test_labels = tf.one_hot(test_labels, num_classes)
        train_images = preprocess_images(train_images)
        test_images = preprocess_images(test_images)
        #train_images = train_images.transpose().reshape(28, 28)
        #test_images = test_images.transpose().reshape(28, 28)

    elif task.lower() == 'bmnist':
        num_classes = 10
        #(train_images, train_labels), (test_images, test_labels) = tfds.as_numpy(tfds.load('binarized_mnist',
        train_images, test_images = tfds.as_numpy(tfds.load('binarized_mnist',
                         split = ['train', 'test'],
                         batch_size=-1))

        #train_labels = tf.one_hot(train_labels, num_classes)
        #test_labels = tf.one_hot(test_labels, num_classes)
        #train_images = preprocess_images(train_images)
        #test_images = preprocess_images(test_images)
    # elif task.lower() == 'cifar10':
    #     num_classes = 10
    #     resize_transform = transforms.Compose([transforms.Resize((28, 28)),
    #                                            transforms.ToTensor()])
    #     transform = transforms.Compose([
    #         transforms.ToTensor()
    #     ])
    #     # transforms.ToTensor() scales input images
    #     # to 0-1 range
    #
    #     train_images = datasets.CIFAR10(root='../../datasets',
    #                                    train=True,
    #                                    transform=transform,
    #                                    download=True)
    #
    #     test_images = datasets.CIFAR10(root='../../datasets',
    #                                   train=False,
    #                                   transform=transform)
    #
    #     train_loader = DataLoader(dataset=train_images,
    #                               batch_size=64,
    #                               shuffle=True)
    #
    #     test_loader = DataLoader(dataset=test_images,
    #                              batch_size=64,
    #                              shuffle=False)
    #
    #     for images, labels in train_loader:
    #         #train_labels = tf.one_hot(labels, num_classes)
    #         train_labels = labels
    #     for images, labels in test_loader:
    #         #test_labels = tf.one_hot(labels, num_classes)
    #         test_labels = labels
    #
    #     train_images = resize_transform(train_images)
    #     test_images = resize_transform(test_images)

    # elif task.lower() == 'kmnist':
    #     num_classes = 10
    #     (train_images, train_labels), (test_images, test_labels) = torchvision.datasets.KMNIST\
    #         (root=r'C:\Users\neloy\OneDrive\GitHub\Thesis\Disentangled-Conditional-Variational-'
    #               r'Autoencoder\dcvae\datasets',download=True)
    #
    #     #kmnist = tfds.load(name="kmnist", split=None)
    #     #kmnist_train = kmnist['train']
    #     #kmnist_test = kmnist['test']
    #     #(train_images, train_labels), (test_images, test_labels) = (kmnist['train']), (kmnist['test'])
    #
    #     train_labels = tf.one_hot(train_labels, num_classes)
    #     test_labels = tf.one_hot(test_labels, num_classes)
    #     train_images = preprocess_images(train_images)
    #     test_images = preprocess_images(test_images)

    elif task.lower() == 'cifar10':
        num_classes = 10
        (train_images, train_labels), (test_images, test_labels) = tfds.as_numpy(tfds.load('cifar10',
                         split = ['train[:9%]', 'test'],
                         batch_size=-1,
                         as_supervised=True))

        train_labels = tf.one_hot(train_labels, num_classes)
        test_labels = tf.one_hot(test_labels, num_classes)
        train_images = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in train_images])
        test_images = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in test_images])
        train_images = preprocess_images_cifar10(train_images)
        test_images = preprocess_images_cifar10(test_images)

    train_dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_types={'x': tf.float32, 'y': tf.float32},
        args=(train_images, train_labels)
    ).shuffle(train_size).take(train_size).batch(batch_size)

    test_dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_types={'x': tf.float32, 'y': tf.float32},
        args=(test_images, test_labels)
    ).shuffle(test_size).take(test_size).batch(test_size)

    return train_dataset, test_dataset


def get_dataset_params(task):
    if task.lower() == 'mnist':
        input_dims = (28, 28, 1)
        kernel_size = (3, 3)
        strides = (2, 2)
    elif task.lower() == 'fmnist':
        input_dims = (28, 28, 1)
        kernel_size = (3, 3)
        strides = (2, 2)
    elif task.lower() == 'cifar10':
        input_dims = (32, 32, 1)
        kernel_size = (3, 3)
        strides = (2, 2)
    elif task.lower() == 'emnist':
        input_dims = (28, 28, 1)
        kernel_size = (3, 3)
        strides = (2, 2)

    elif task.lower() == 'kmnist':
        input_dims = (28, 28, 1)
        kernel_size = (3, 3)
        strides = (2, 2)
    elif task.lower() == 'bmnist':
        input_dims = (28, 28, 1)
        kernel_size = (3, 3)
        strides = (2, 2)

    return input_dims, kernel_size, strides