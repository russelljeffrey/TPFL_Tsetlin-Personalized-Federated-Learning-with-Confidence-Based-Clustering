import yaml
import argparse
from server import Server
from client import Client
from model import create_model
from data_loader import partition_data
from preprocessing import binarize_and_reshape

def main(selected_experiment, selected_dataset):
    with open("config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)
    
    num_clients = config['num_clients']
    num_rounds = config['num_rounds']
    fraction_fit = config['fraction_fit']
    clause = config['clause']
    T = config['T']
    sensitivity = config['sensitivity']
    experiments = config['experiments']
    
    experiment_config = experiments[str(selected_experiment)]
    iid_percent = experiment_config['iid_percent']
    non_iid_percent = experiment_config['non_iid_percent']
    num_iid_clients = int(num_clients * iid_percent / 100)
    num_non_iid_clients = int(num_clients * non_iid_percent / 100)
    server = Server(num_rounds=num_rounds, fraction_fit=fraction_fit)

    client_id = 0
    # IID clients
    for _ in range(num_iid_clients):
        client_model = create_model(clause, T, sensitivity)
        train_set, test_set, confidence_set = partition_data(True, selected_dataset)
        x_train, y_train = train_set
        x_test, y_test = test_set
        x_confidence, y_confidence = confidence_set

        x_train = binarize_and_reshape(x_train, x_train.shape)
        x_test = binarize_and_reshape(x_test, x_test.shape)
        x_confidence = binarize_and_reshape(x_confidence, x_confidence.shape)

        dataset = (x_train, y_train, x_test, y_test, x_confidence)
        print(f"IID Client {client_id + 1} Preprocessed")
        client = Client(client_id=client_id, dataset=dataset, model=client_model)
        server.add_client(client)
        client_id += 1

    # non-IID clients
    for _ in range(num_non_iid_clients):
        client_model = create_model(clause, T, sensitivity)
        train_set, test_set, confidence_set = partition_data(False, selected_dataset)
        x_train, y_train = train_set
        x_test, y_test = test_set
        x_confidence, y_confidence = confidence_set

        x_train = binarize_and_reshape(x_train, x_train.shape)
        x_test = binarize_and_reshape(x_test, x_test.shape)
        x_confidence = binarize_and_reshape(x_confidence, x_confidence.shape)

        dataset = (x_train, y_train, x_test, y_test, x_confidence)
        print(f"Non-IID Client {client_id + 1} Preprocessed")
        client = Client(client_id=client_id, dataset=dataset, model=client_model)
        server.add_client(client)
        client_id += 1

    server.start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Experiment Runner")
    parser.add_argument("--experiment", type=int, required=True, help="Experiment number to run (1-5)")
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "fashion_mnist", "cifar10", "emnist"], help="Dataset to use")
    args = parser.parse_args()
    
    main(args.experiment, args.dataset)