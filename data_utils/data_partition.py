import os
import json
import os
import numpy as np
from scipy.stats import dirichlet
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from interval3 import Interval
import copy
import pickle
def partition(data_path, save_path, num_clients, dirichlet_alpha, partition_method="dirichlet_label", num_of_classes_for_stsb=5):

    data_path2 = save_path
    df = pd.read_json(data_path, orient='records')
    dataset_len = len(df)
    if "sts-b" in data_path:
        unique_label_list, num_unique_labels = generate_stsB_dataframe(df, num_of_classes=num_of_classes_for_stsb)
    else:
        unique_label_list = np.array(df['response'].unique())
        num_unique_labels = len(df['response'].unique())
    
    if partition_method == "dirichlet_label_uni":
        p = 1
        num_classes = num_unique_labels
        # Phi = 10 * 2
        Phi = np.random.binomial(1, p, size=(num_clients, num_classes))  # indicate the classes chosen by each client
        n_classes_per_client = np.sum(Phi, axis=1)
        while np.min(n_classes_per_client) == 0:
            invalid_idx = np.where(n_classes_per_client == 0)[0]
            Phi[invalid_idx] = np.random.binomial(1, p, size=(len(invalid_idx), num_classes))
            n_classes_per_client = np.sum(Phi, axis=1)
        #     Psi = 10 * 2
        Psi = [list(np.where(Phi[:, j] == 1)[0]) for j in range(num_classes)]  # indicate the clients that choose each class
        num_clients_per_class = np.array([len(x) for x in Psi])
        if "sts-b" in data_path:
            y_train = np.array(df['label']) 
        else:
            y_train = np.array(df['response'])
        dict_users = {}
        for class_i, class_name in enumerate(unique_label_list):
            # all index of the class_i
            all_idxs = np.where(y_train == class_name)[0]
            p_dirichlet = np.random.dirichlet([dirichlet_alpha] * num_clients_per_class[class_i])
            # produce a list, each element of the list indicates which client it belongs
            assignment = np.random.choice(Psi[class_i], size=len(all_idxs), p=p_dirichlet.tolist())

            # for all the clients who have the training sample of class_i
            for client_k in Psi[class_i]:
                if client_k in dict_users:
                    # union set
                    dict_users[client_k] = set(dict_users[client_k] | set(all_idxs[(assignment == client_k)]))
                else:
                    dict_users[client_k] = set(all_idxs[(assignment == client_k)])
        # print(dict_users)
        os.makedirs(data_path2, exist_ok=True)
        num_for_each_client = []
        for client_id in range(num_clients):
            print(
                "\n Generating the local training dataset of Client_{}".format(client_id)
            )
            sub_df = df.loc[list(dict_users[client_id])]
            sub_df = sub_df.reset_index().drop('index', axis=1)
            num_for_each_label = generate_num_for_each_label_per_client(data_path, sub_df, unique_label_list)
            num_for_each_client.append(num_for_each_label)   
            sub_remaining_df_dic = sub_df.to_dict(orient='records')
            with open(os.path.join(data_path2, "local_training_{}.json".format(client_id)), 'w') as outfile:
                json.dump(sub_remaining_df_dic, outfile, indent=2)
            # num_for_each_client: number of client * number of labels
        visualize(num_for_each_client, num_clients, data_path2, dirichlet_alpha, unique_label_list, partition_method)
        with open(os.path.join(data_path2, "heterogeneity.pkl"), 'wb') as f:
            pickle.dump(num_for_each_client, f)
        return num_for_each_client
    
    elif partition_method == "dirichlet_label":
        if "sts-b" in data_path:
            all_label_list = np.array(df['label'])
        else:
            all_label_list = np.array(df['response'])
        # set alpha = 100 means that the num of samples are nearly uniformly distributed among clients
        dirichlet_samples = dirichlet.rvs([100] * num_clients, size=1)
        num_samples_per_client = (np.floor(dirichlet_samples * dataset_len).astype(int)).squeeze()
        print(num_samples_per_client)
        cls_priors = np.random.dirichlet(alpha=[dirichlet_alpha] * num_unique_labels, size=num_clients)
        # print(cls_priors)
        prior_cumsum = np.cumsum(cls_priors, axis=1)
        idx_list = [np.where(all_label_list == i)[0] for i in unique_label_list]
        cls_amount = [len(idx_list[i]) for i in list(range(len(idx_list)))]
        original_cls_amount = copy.deepcopy(cls_amount)
        sample_idx_per_client = [[] for _ in range(num_clients)]
        while np.sum(num_samples_per_client) != 0:
            # print("sum of samples per client: " + str(np.sum(num_samples_per_client)))
            curr_clnt = np.random.randint(num_clients)
            if num_samples_per_client[curr_clnt] <= 0:
                continue
            num_samples_per_client[curr_clnt] -= 1
            curr_prior = prior_cumsum[curr_clnt]
            while True:
                cls_label = np.argmax(np.random.uniform() <= curr_prior)
                # Redraw class label if trn_y is out of that class
                if cls_amount[cls_label] <= 0:
                    cls_amount[cls_label] = original_cls_amount[cls_label]
                    # continue
                cls_amount[cls_label] -= 1
                sample_idx_per_client[curr_clnt].append(idx_list[cls_label][cls_amount[cls_label]])
                break
        
        os.makedirs(data_path2, exist_ok=True)

        num_for_each_client = []

        for client_id in range(num_clients):
            print(
                "\n Generating the local training dataset of Client_{}".format(client_id)
            )
            sub_df = df.loc[sample_idx_per_client[client_id]]
            sub_df = sub_df.reset_index().drop('index', axis=1)
            num_for_each_label = generate_num_for_each_label_per_client(data_path, sub_df, unique_label_list)
            num_for_each_client.append(num_for_each_label)   
            sub_remaining_df_dic = sub_df.to_dict(orient='records')
            with open(os.path.join(data_path2, "local_training_{}.json".format(client_id)), 'w') as outfile:
                json.dump(sub_remaining_df_dic, outfile, indent=2)
        visualize(num_for_each_client, num_clients, data_path2, dirichlet_alpha, unique_label_list, partition_method)
    
    elif partition_method == "dirichlet_quantity":
        # df = pd.read_json(data_path, orient='records')
        # dataset_len = len(df)
        # num_unique_labels = len(df['response'].unique())
        dirichlet_samples = dirichlet.rvs([dirichlet_alpha] * num_clients, size=1)
        client_samples = (np.floor(dirichlet_samples * dataset_len).astype(int)).squeeze()
        
        os.makedirs(data_path2, exist_ok=True)
        num_for_each_client = []

        for client_id, size in enumerate(client_samples):
            print(
                "\n Generating the local training dataset of Client_{}".format(client_id)
            )
            sub_df = df.sample(frac=(size / len(df)))
            df = df.drop(sub_df.index)
            sub_df = sub_df.reset_index().drop('index', axis=1)
            num_for_each_label = generate_num_for_each_label_per_client(data_path, sub_df, unique_label_list)
            num_for_each_client.append(num_for_each_label)
            sub_remaining_df_dic = sub_df.to_dict(orient='records')
            with open(os.path.join(data_path2, "local_training_{}.json".format(client_id)), 'w') as outfile:
                json.dump(sub_remaining_df_dic, outfile, indent=2)

        visualize(num_for_each_client, num_clients, data_path2, dirichlet_alpha, unique_label_list, partition_method)
    
    elif partition_method == 'iid':
        # df = pd.read_json(data_path, orient='records')
        # dataset_len = len(df)
        # num_unique_labels = len(df['response'].unique())
        dirichlet_samples = dirichlet.rvs([10000] * num_clients, size=1)
        client_samples = (np.floor(dirichlet_samples * dataset_len).astype(int)).squeeze()
        
        os.makedirs(data_path2, exist_ok=True)
        num_for_each_client = []

        for client_id, size in enumerate(client_samples):
            print(
                "\n Generating the local training dataset of Client_{}".format(client_id)
            )
            sub_df = df.sample(frac=(size / len(df)))
            df = df.drop(sub_df.index)
            sub_df = sub_df.reset_index().drop('index', axis=1)
            num_for_each_label = generate_num_for_each_label_per_client(data_path, sub_df, unique_label_list)
            num_for_each_client.append(num_for_each_label)
            sub_remaining_df_dic = sub_df.to_dict(orient='records')
            with open(os.path.join(data_path2, "local_training_{}.json".format(client_id)), 'w') as outfile:
                json.dump(sub_remaining_df_dic, outfile, indent=2)

        visualize(num_for_each_client, num_clients, data_path2, dirichlet_alpha, unique_label_list, partition_method)

def visualize(num_for_each_client, num_clients, data_path2, dirichlet_alpha, unique_label_list, partition_method):
    # Transpose operation
    plt.figure(figsize=(15, 6))
    num_for_each_label_each_client = list(map(list, zip(*num_for_each_client)))
    x = np.arange(num_clients)
    unique_label_list.sort()
    for index, label in enumerate(unique_label_list):
        if index == 0:
            bottom_height = [0]*num_clients
            plt.bar(x, num_for_each_label_each_client[index], label=label)
            # for a, b in zip(x, num_for_each_label_each_client[index]):
                # 柱子上的数字显示
                # plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
        else:
            bottom_height = [i + j for i, j in zip(bottom_height, num_for_each_label_each_client[index-1])]
            plt.bar(x, num_for_each_label_each_client[index], bottom=bottom_height, label=label)
            bottom_height_for_text = [i + j for i, j in zip(bottom_height, num_for_each_label_each_client[index])]
            # for a, b in zip(x, bottom_height_for_text):
                # 柱子上的数字显示
                # plt.text(a, b, b, ha='center', va='bottom', fontsize=10)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Client", fontsize=12)
    plt.ylabel("Total sample number", fontsize=12)
    if partition_method == 'dirichlet_quantity':
        plt.title("dirichlet quantity, alpha = " + str(dirichlet_alpha))
    elif partition_method == 'dirichlet_label':
        plt.title("dirichlet label, alpha = " + str(dirichlet_alpha))
    elif partition_method == "dirichlet_label_uni":
        plt.title("dirichlet label uniform, alpha = " + str(dirichlet_alpha))
    elif partition_method == 'iid':
        plt.title("iid")
    plt.legend(bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)
    plt.savefig(os.path.join(data_path2, "disrtibution.png"))


def generate_num_for_each_label_per_client(datapath, subdf, unique_label_list):
        # 可视化分布步骤
        if "sts-b" in datapath:
            num_each_label = subdf['label'].value_counts()
            unique_label_client_list = np.array(subdf['label'].unique())
        else:
            num_each_label = subdf['response'].value_counts()
            unique_label_client_list = np.array(subdf['response'].unique())
        unique_label_client_list.sort()
        # 如果client有部分标签不存在，则以下代码可以解决可视化中的问题
        # 算出 unique_label_client_list和 unique_label_list 差那些label，然后在unique_label_list找到索引，然后在num_for_each_label补0
        unique_label_list.sort()
        unique_label_list = list(unique_label_list)
        difference = list(set(unique_label_list) - set(unique_label_client_list))
        difference.sort()
        differnence_indexes = [unique_label_list.index(item) for item in difference]
        # 如果client有部分标签不存在，则以上代码可以解决可视化中的问题
        num_for_each_label = []
        for index, label in enumerate(unique_label_client_list):
            num = num_each_label[label]
            num_for_each_label.append(num)
        # 如果client有部分标签不存在，则以下代码可以解决可视化中的问题
        for difference_index in differnence_indexes:
            num_for_each_label.insert(difference_index, 0)
        # 如果client有部分标签不存在，则以上代码可以解决可视化中的问题

        return num_for_each_label


def generate_stsB_dataframe(stsb, num_of_classes):
    stsb.insert(stsb.shape[1], 'label', 0)
    gap = 5 / num_of_classes
    left = 0
    intervals = []
    for _ in range(num_of_classes):
        # 左闭右开
        if left + gap == 5:
            new_interval = Interval(left, left+gap)
        else:
            new_interval = Interval(left, left+gap, upper_closed=False)
        left += gap
        intervals.append(new_interval)
    
    for index, row in stsb.iterrows():
        for id, interval in enumerate(intervals):
            if(row['response'] in interval):
                stsb.loc[index,'label'] = id
                break
    unique_label_list = np.array(stsb['label'].unique())
    num_unique_labels = len(stsb['label'].unique())
    return unique_label_list, num_unique_labels

class DataPartition:
    def __init__(self, args) -> None:
        self.dataset = args.dataset
        self.num_client = args.num_clients
        self.dirichlet_alpha = args.dirichlet_alpha
        self.partition_method = args.partition_method
        self.data_path = args.data_path

    def partition(self):
        data_folders = {
            "20news": "/home/jianhuiwei/rsch/jianhui/dataset/20news/train.json",
            # "20news": "/data/jianhui/dataset/20news/train.json",
        }
        data_folder = data_folders[self.dataset]
        if self.dataset == "20news":
            return partition(data_path=data_folder, save_path=self.data_path,num_clients=self.num_client, dirichlet_alpha=self.dirichlet_alpha,
                      partition_method=self.partition_method)
            
def partition_data(args):
    data_partition = DataPartition(args)
    return data_partition.partition()