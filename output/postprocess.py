
import pandas as pd
import matplotlib.pyplot as plt
import os

def draw_acc_curves(short_result_file_names):
    result_list = []
    indexes = []
    for file_name in short_result_file_names:
        result = pd.read_csv(
        file_name,
        header=None,
        sep=' ',
        )
        index = result.iloc[:, 0].tolist()
        indexes.append(index)
        acc_list = result.iloc[:, 1].tolist()
        result_list.append(acc_list)
    
    method_title = ['FedAvg', 'FedAvgM-momentum_0.2', 'FedProx-'+chr(956)+'_0.001', 'Scaffold-lr_0.5']
    method_title = ['FedAvg', 'FedProx', 'FedAvgM', 'Scaffold', 'Ours']

    for id, (index, result) in enumerate(zip(indexes, result_list)):
        plt.plot(index, result, alpha=0.5, linewidth=1, label=method_title[id])

    plt.legend()  #显示上面的label
    plt.xlabel('Rounds')
    plt.ylabel('Accurcy') #accuracy
    plt.title('Accuracy curve for heterogeneity at ' + chr(945) + " = 0.1")
    save_path = os.path.join(os.path.dirname(short_result_file_names[0]), 'Accuracy_curve.png')
    plt.savefig(save_path)

if __name__ == "__main__":
    # response_path = 'output/quail-dirichlet_label_alpha=1-imbalanced_label-alpaca-lora/19.xlsx'
    # process_result_from_quail(response_path)
    short_result_file_name = 'output/20news-roberta-lora-dirichlet_label_uni-0.05-10-FedProx-0.1/short_result.txt'
    short_result_file_name2 = 'output/20news-roberta-lora-dirichlet_label_uni-0.05-10-NewMethod/std 0.8 weight.txt'
    short_result_file_name3 = 'output/20news-roberta-lora-dirichlet_label_uni-0.05-10-NewMethod/short_result.txt'
    short_result_file_name4 = 'output/20news-roberta-lora-dirichlet_label_uni-0.1-10-FedProx-0.07-method1+2+3/short_result.txt'
    short_result_file_name5 = 'output/20news-roberta-lora-dirichlet_label_uni-0.1-10-FedProx-0.07-method1+2+3/short_result.txt'
    # short_result_file_name5 = 'output/20news-roberta-lora-dirichlet_label_uni-0.1-10-FedProx-0.005/short_result.txt'
    file_list = [short_result_file_name, short_result_file_name2]
    draw_acc_curves(file_list)