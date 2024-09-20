from dataclasses import dataclass
import tensorflow as tf


@dataclass
class FMNISTDataset:
    name: str
    lossfn: tf.Tensor
    train_images: tf.Tensor
    train_labels: tf.Tensor
    validation_images: tf.Tensor
    validation_labels: tf.Tensor
    test_images: tf.Tensor
    test_labels: tf.Tensor

def preprocess_dataset():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

    # normalizing data
    train_images, test_images = train_images / 255.0, test_images / 255.0

    validation_images, validation_labels = test_images[0:5000], test_labels[0:5000]
    test_images, test_labels = test_images[5000:], test_labels[5000:]
    
    # lossfn
    lossfn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    print("FMNIST")

    return FMNISTDataset("FMNIST", lossfn, train_images, train_labels, validation_images, validation_labels, test_images, test_labels)