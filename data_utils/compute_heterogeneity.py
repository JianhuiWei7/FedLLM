import numpy as np

def compute_heterogeneity(num_for_each_client):
    std_for_each_client = []
    relative_heterogeneity_for_each_client = []
    for item in num_for_each_client:
        total_number = np.sum(item)
        std_for_each_client.append(np.std(item) ** 1.2)
        relative_heterogeneity_for_each_client.append(np.std(item) ** 0.6)
    return relative_heterogeneity_for_each_client, std_for_each_client



if __name__ == "__main__":
    num_for_each_client = [
        [10,10,10,10,10],
        [10,0,10,10,10]
    ]
    print(compute_heterogeneity(num_for_each_client))
