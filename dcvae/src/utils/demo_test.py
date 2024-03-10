import tempfile
import torchvision
from src.utils.data_loader import *
import numpy as np

def preprocess_images_cifar10(images, dims=(-1, 28, 28, 1)):
    images = images.reshape(dims) / 255.
    return np.where(images > .5, 1.0, 0.0).astype('float32')


def get_dataset(task, train_size, test_size, batch_size):
    num_classes = 10
    (train_images, train_labels), (test_images, test_labels)  = torchvision.datasets.CIFAR10(
            root=tempfile.gettempdir(),
            download=True,
            train=True,
            # Simply put the size you want in Resize (can be tuple for height, width)
            transform=torchvision.transforms.Compose([torchvision.transforms.Resize(28),
                                                      torchvision.transforms.ToTensor()]
    ),
)

    train_labels = tf.one_hot(train_labels, num_classes)
    test_labels = tf.one_hot(test_labels, num_classes)
    train_images = preprocess_images_cifar10(train_images)
    test_images = preprocess_images_cifar10(test_images)

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

    #return train_dataset, test_dataset
    print(train_dataset[0][0].shape) # 1, 32, 32 (channels, width, height)
    print(train_dataset[0][1].shape) # 1, 10 (channels, width, height)
    print(test_dataset[0][0].shape) # 1, 32, 32 (channels, width, height)
    print(test_dataset[0][1].shape) # 1, 10 (channels, width, height)
