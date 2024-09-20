import tensorflow_probability as tfp
import numpy as np
from keras.datasets import mnist, fashion_mnist
from torchvision import datasets, transforms

def shuffle_data(data: np.ndarray, labels: np.ndarray):
    """
    This function shuffles the data and labels while preserving the correspondence between them.

    Parameters:
        data (numpy.ndarray): The dataset to be shuffled.
        labels (numpy.ndarray): The labels corresponding to the dataset.

    Returns:
        tuple: Shuffled data and labels as two numpy arrays.
    """
    assert data.shape[0] == labels.shape[0], "Data and labels must have the same number of samples"

    permutation = np.random.permutation(data.shape[0])

    shuffled_data = data[permutation]
    shuffled_labels = labels[permutation]

    return shuffled_data, shuffled_labels

def partition_data_single_client(iid, images, labels, num_classes, samples_per_client, alpha, available_indices):
    """
    This function partitions data for a single client based on a Dirichlet distribution.
    Only the last class is guaranteed to have at least one sample. This is due to Tsetlin Machine Nature.
    """
    if isinstance(alpha, float):
        alpha = np.full(num_classes, alpha)

    dirichlet_dist = tfp.distributions.Dirichlet(alpha)
    class_probs = dirichlet_dist.sample(1).numpy().reshape(-1)

    class_counts = np.round(class_probs * samples_per_client).astype(int)

    # At least one sample for the last class
    if class_counts[-1] == 0:
        class_counts[-1] = 1

    total_count = np.sum(class_counts)
    if total_count != samples_per_client:
        diff = samples_per_client - total_count
        class_counts[np.argmax(class_counts)] += diff

    label_to_indices = {}
    for label in range(num_classes):
        label_indices = np.where(labels == label)[0]
        available_class_indices = np.intersect1d(label_indices, available_indices)
        np.random.shuffle(available_class_indices)
        label_to_indices[label] = available_class_indices

    client_data_partitioned = []
    client_labels_partitioned = []

    for label in range(num_classes):
        count = class_counts[label]
        if label in label_to_indices and label_to_indices[label].size > 0:
            selected_indices = label_to_indices[label][:count]
            client_data_partitioned.append(images[selected_indices])
            client_labels_partitioned.append(labels[selected_indices])

            if len(selected_indices) < count:
                shortfall = count - len(selected_indices)
                if len(selected_indices) > 0:
                    resampled_indices = np.random.choice(selected_indices, shortfall, replace=True)
                    client_data_partitioned.append(images[resampled_indices])
                    client_labels_partitioned.append(labels[resampled_indices])

    client_data_partitioned = np.concatenate(client_data_partitioned)
    client_labels_partitioned = np.concatenate(client_labels_partitioned)

    return client_data_partitioned, client_labels_partitioned, set()

def load_emnist():
    transform = transforms.Compose([
    transforms.ToTensor()
    ])

    x_train = datasets.EMNIST(root='./data', split='byclass', train=True, download=True, transform=transform)
    x_test = datasets.EMNIST(root='./data', split='byclass', train=False, download=True, transform=transform)

    train_data = x_train.data.numpy()
    train_labels = x_train.targets.numpy()
    test_data = x_test.data.numpy()
    test_labels = x_test.targets.numpy()

    x = np.concatenate((train_data, test_data), axis=0)
    y = np.concatenate((train_labels, test_labels), axis=0)

    return x, y

def load_dataset(dataset):
    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    
    return x, y

def sub_sample_set(images, labels, sample_size):
    indices = np.random.choice(len(images), sample_size, replace=False)
    sampled_data = images[indices]
    sampled_labels = labels[indices]
    return sampled_data, sampled_labels

def partition_data(iid: bool, dataset: str):
    """
    This function partitions data based on Dirichlet distribution using partition_data_single_client function

    Args:
        iid (bool): an attribute to define whether data should be partitioned in an iid or non-iid fashion
        if iid is set to True, alpha value is 10000.0 (iid data), if set to False, alpha value is 0.05 (non-iid data).
        images (numpy.ndarray): The dataset to be partitioned.
        labels (numpy.ndarray): The labels corresponding to the dataset.
        dataset (str): the desired dataset for clients.

    Returns:
        a tuple containing three tuples >>> (train_data, train_labels), (test_data, test_labels), (confidence_data, confidence_labels)
        Each tuple has the data and labels partitioned for a single client
    """
    if dataset == "mnist":
        images, labels = load_dataset(mnist)
    elif dataset == "fashion_mnist":
        images, labels = load_dataset(fashion_mnist)
    elif dataset == "emnist":
        images, labels = load_emnist()

    images, labels = shuffle_data(images, labels)

    num_samples = len(images)
    num_classes = len(np.unique(labels))
    alpha = np.full(num_classes, 10000.0 if iid else 0.05)

    samples_per_client = 30000

    available_indices = np.arange(num_samples)

    train_data, train_labels, used_indices = partition_data_single_client(iid, images, labels, num_classes, samples_per_client, alpha, available_indices)

    test_set_size = int(0.5 * len(train_data))
    confidence_set_size = int(0.5 * len(train_data))
    test_data, test_labels = sub_sample_set(train_data, train_labels, test_set_size)
    confidence_data, confidence_labels = sub_sample_set(train_data, train_labels, confidence_set_size)

    return (train_data, train_labels), (test_data, test_labels), (confidence_data, confidence_labels)