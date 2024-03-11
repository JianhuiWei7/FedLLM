import numpy as np

def compute_heterogeneity(num_for_each_client):
    heterogeneity_for_each_client = []
    for item in num_for_each_client:
        heterogeneity_for_each_client.append(np.std(item))
    return heterogeneity_for_each_client



if __name__ == "__main__":
    num_for_each_client = [
        [10,10,10,10,10],
        [10,0,10,10,10]
    ]
    print(compute_heterogeneity(num_for_each_client))
