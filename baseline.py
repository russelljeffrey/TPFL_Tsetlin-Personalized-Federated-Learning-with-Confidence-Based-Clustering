import argparse
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.datasets import mnist, fashion_mnist
import tensorflow_probability as tfp
from torchvision import datasets, transforms

# Functions for partitioning data
def shuffle_data(data: np.ndarray, labels: np.ndarray):
    permutation = np.random.permutation(data.shape[0])
    shuffled_data = data[permutation]
    shuffled_labels = labels[permutation]
    return shuffled_data, shuffled_labels

def partition_data_single_client(iid, images, labels, num_classes, samples_per_client, alpha, available_indices):
    if isinstance(alpha, float):
        alpha = np.full(num_classes, alpha)

    dirichlet_dist = tfp.distributions.Dirichlet(alpha)
    class_probs = dirichlet_dist.sample(1).numpy().reshape(-1)
    class_counts = np.round(class_probs * samples_per_client).astype(int)

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

def load_dataset(dataset):
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    return x, y

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

def sub_sample_set(images, labels, sample_size):
    indices = np.random.choice(len(images), sample_size, replace=False)
    sampled_data = images[indices]
    sampled_labels = labels[indices]
    return sampled_data, sampled_labels

def partition_data(iid: bool, dataset: str):
    if dataset == "mnist":
        images, labels = load_dataset(mnist)
    elif dataset == "fashion_mnist":
        images, labels = load_dataset(fashion_mnist)
    elif dataset == "emnist":
        images, labels = load_emnist()

    images, labels = shuffle_data(images, labels)
    num_samples = len(images)
    num_classes = len(np.unique(labels))
    alpha = np.full(num_classes, 10000.0 if iid else 0.05) # 0.05 for non-iid and 10000 for iid
    samples_per_client = 30000
    available_indices = np.arange(num_samples)

    train_data, train_labels, used_indices = partition_data_single_client(
        iid, images, labels, num_classes, samples_per_client, alpha, available_indices
    )

    test_set_size = int(0.5 * len(train_data))
    test_data, test_labels = sub_sample_set(train_data, train_labels, test_set_size)

    return (train_data, train_labels), (test_data, test_labels)

# Model-related code
def create_cnn_model():
    return models.Sequential([
        layers.Conv2D(20, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(50, kernel_size=(5, 5), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(500, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

def train_client_model(model, train_data, train_labels, epochs=10, mu=0, global_weights=None):
    if global_weights:
        model.set_weights(global_weights)
    
    optimizer = optimizers.SGD(learning_rate=0.01)
    dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(32)
    
    total_loss = 0
    num_batches = 0
    
    for epoch in range(epochs):
        for features, labels in dataset:
            with tf.GradientTape() as tape:
                predictions = model(features, training=True)
                loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, predictions))
                if mu > 0 and global_weights:
                    loss += mu / 2 * sum(tf.reduce_sum(tf.square(w1 - w2)) 
                                         for w1, w2 in zip(model.trainable_weights, global_weights))
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            
            total_loss += loss.numpy()
            num_batches += 1
    avg_loss = total_loss / num_batches
    return model.get_weights(), avg_loss

def evaluate_model(model, test_data, test_labels):
    dataset = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(32)
    correct = 0
    total = 0
    for features, labels in dataset:
        predictions = model(features, training=False)
        correct += np.sum(np.argmax(predictions, axis=1) == labels)
        total += labels.shape[0]
    return 100 * correct / total

def main(dataset_name, strategy):
    num_clients = 10
    mu = 0.01 if strategy == 'fedprox' else 0

    global_model = create_cnn_model()
    global_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    training_sets = []
    testing_sets = []

    for _ in range(num_clients):
        (train_data, train_labels), (test_data, test_labels) = partition_data(iid=False, dataset=dataset_name)
        train_data = train_data.reshape(-1, 28, 28, 1) / 255.0
        test_data = test_data.reshape(-1, 28, 28, 1) / 255.0
        training_sets.append((train_data, train_labels))
        testing_sets.append((test_data, test_labels))

    # Use GPU if available
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

    with tf.device(device):
        for round_num in range(10):
            client_weights = []
            client_losses = []
            total_upload_cost = 0

            for train_data, train_labels in training_sets:
                client_model = create_cnn_model()
                weights, loss = train_client_model(client_model, train_data, train_labels, mu=mu, global_weights=global_model.get_weights())
                client_weights.append(weights)
                client_losses.append(loss)
                total_upload_cost += sum(np.prod(w.shape) for w in weights)
            
            new_weights = [np.mean([client_weights[j][i] for j in range(num_clients)], axis=0) for i in range(len(client_weights[0]))]
            global_model.set_weights(new_weights)

            total_download_cost = sum(np.prod(w.shape) for w in new_weights)
            avg_loss = np.mean(client_losses)
            accuracy = evaluate_model(global_model, testing_sets[0][0], testing_sets[0][1])

            print(f"Round {round_num + 1}:")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Accuracy: {accuracy:.2f}%")
            print(f"  Upload Cost: {total_upload_cost} elements")
            print(f"  Download Cost: {total_download_cost} elements")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Learning Simulation')
    parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'fashion_mnist', 'emnist'], help='Dataset to use')
    parser.add_argument('--strategy', type=str, required=True, choices=['fedavg', 'fedprox'], help='Federated learning strategy to use')
    args = parser.parse_args()
    main(args.dataset, args.strategy)