import numpy as np
from collections import defaultdict
import server_logging as log

class Server:
    def __init__(self, num_rounds, fraction_fit):
        self.num_rounds = num_rounds
        self.fraction_fit = fraction_fit
        self.round = 0
        self.accuracies = []
        self.upload_costs = []
        self.download_costs = []
        self.clients = []
        self.clusters = defaultdict(list) 

    def add_client(self, client):
        self.clients.append(client)

    def aggregate_fit(self, results):
        new_clusters = defaultdict(list)
        client_accuracies = []
        upload_cost = 0

        for client_id, (parameters, num_examples, metrics) in results.items():
            if parameters:
                class_label, weights = parameters
                new_clusters[class_label].append(weights)
                if 'accuracy' in metrics:
                    client_accuracies.append(metrics['accuracy'])
                upload_cost += weights.nbytes

        aggregated_parameters = {}
        download_cost = 0
        for class_label, weights_list in new_clusters.items():
            weights_list = [np.array(w) for w in weights_list]
            if class_label in self.clusters:
                current_cluster_weights = np.vstack(self.clusters[class_label])
                new_cluster_weights = np.vstack(weights_list)
                combined_weights = np.vstack([current_cluster_weights, new_cluster_weights])
                aggregated_weights = np.mean(combined_weights, axis=0)
            else:
                aggregated_weights = np.mean(weights_list, axis=0)

            aggregated_parameters[class_label] = (class_label, aggregated_weights)
            download_cost += aggregated_weights.nbytes
            self.clusters[class_label] = [aggregated_weights]

        round_accuracy = np.mean(client_accuracies) if client_accuracies else float('nan')
        self.upload_costs.append(upload_cost)
        self.download_costs.append(download_cost)
        self.round += 1

        log.log_upload(self.round, self.upload_costs)
        log.log_download(self.round, self.download_costs)

        return aggregated_parameters, round_accuracy

    def evaluate(self):
        accuracies = [client.evaluate() for client in self.clients]
        avg_accuracy = np.mean(accuracies)
        self.accuracies.append(avg_accuracy)
        log.log_accuracy(self.round, self.accuracies)
        log.log_round(self.round, avg_accuracy, self.clusters)  # Log information
        return avg_accuracy

    def start(self):
        for round_num in range(1, self.num_rounds + 1):
            print(f"Round {round_num} started")

            # Step 1: Fitting local data
            for client in self.clients:
                client.fit()

            # Step 2: Getting clients parameters
            results = {}
            for client in self.clients:
                results[client.client_id] = client.get_parameters(), len(client.x_train), {"accuracy": client.evaluate()}

            # Step 3 and 4: Aggregate parameters on the server - set new parameters on each client
            aggregated_parameters, round_accuracy = self.aggregate_fit(results)

            # Step 5: Set new parameters
            for client in self.clients:
                class_label = client.model_class_confidence
                if class_label in aggregated_parameters:
                    client.set_parameters(aggregated_parameters[class_label])

            # Step 6: Evaluate the entire federation
            avg_accuracy = self.evaluate()

            # Log cluster information
            print(f"End of round {round_num}")
            print()

