import pandas as pd

def log_accuracy(round_num, accuracies):
    accuracy_df = pd.DataFrame({"round": list(range(1, round_num + 1)), "accuracy": accuracies})
    accuracy_df.to_csv("round_accuracies.csv", index=False)

def log_upload(round_num, upload_costs):
    upload_cost_df = pd.DataFrame({"round": list(range(1, round_num + 1)), "upload_cost": upload_costs})
    upload_cost_df.to_csv("upload_costs.csv", index=False)

def log_download(round_num, download_costs):
    download_cost_df = pd.DataFrame({"round": list(range(1, round_num + 1)), "download_cost": download_costs})
    download_cost_df.to_csv("download_costs.csv", index=False)

def log_round(round_num, avg_accuracy, clusters):
    cluster_info = [class_label for class_label in clusters.keys()]
    print(f"Round {round_num} - Average Accuracy: {avg_accuracy:.4f}, Clusters: {cluster_info}")