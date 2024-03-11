import argparse
import os
# lr for roberta-lora: 3e-4
# lr for roberta-p_tuningV2: 8e-3
# lr for roberta-IA3: 3e-3
# lr for bert-lora: 3e-3
# lr for bert-IA3: 3e-3
# lr for bert-p_tuningV2: 8e-3
def parse_args():
    parser = argparse.ArgumentParser(description="Federated Learning PEFine-Tuning for LLM")
    parser.add_argument('--model', type=str, default='roberta', help='which pretrained model to use, now support roberta and bert')
    parser.add_argument('--peft_method', type=str, default='lora', help='which peft method to use, now support lora, p_tuningV2, IA3')
    # parameters for lora adapter
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA r parameter')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout rate')
    parser.add_argument('--lora_target_modules', nargs='+', default=["q_proj", "k_proj", "v_proj", "o_proj"], help='LoRA target modules')
    # parameters for prefix tuning
    parser.add_argument('--num_virtual_tokens', type=int, default=8, help='num of virtual tokens for prefix tuning')
    
    parser.add_argument('--dataset', type=str, default='20news', help='Dataset to use')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.1, help='dirichlet alpha parameter, 1, 1.5, 2')
    parser.add_argument('--partition_method', type=str, default="dirichlet_label_uni", help='The method used to partition the data, choose from [''iid'', ''dirichlet_label_uni'', ''dirichlet_label'', ''dirichlet_quantity'']')
    parser.add_argument('--client_selection_strategy', type=str, default='random', help='Client selection strategy')
    parser.add_argument('--client_selection_frac', type=float, default=0.4, help='Fraction of clients to select')
    parser.add_argument('--num_communication_rounds', type=int, default=80, help='Number of communication rounds')
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients')
    # FedProx related arguments
    parser.add_argument('--useFedProx', type=bool, default=False, help='Whether or not add proximal term to the loss function')
    parser.add_argument('--proximal_term_argument', type=float, default=0.07, help='the mu for proximal term')
    # FedAvgM related arguments
    parser.add_argument('--useFedAvgM', type=bool, default=False, help='Whether or not use FedAvgM for aggregation')
    parser.add_argument('--beta', type=float, default=0.7, help='hyperparameter for FedAvgM beta')
    # Scaffold related arguments
    parser.add_argument('--useScaffold', type=bool, default=False, help='Whether or not use Scaffold')
    
    # parser.add_argument('--scaffold_dir', type=str, default='/data/jianhui/scaffold_control_variate', help='the dir to save variate for server and client')
    parser.add_argument('--scaffold_dir', type=str, default='/home/jianhuiwei/rsch/jianhui/scaffold_control_variate', help='the dir to save variate for server and client')
    
    parser.add_argument('--useHeterogeneityWeight', type=bool, default=True, help='Whether useHeterogeneityWeight')
    parser.add_argument('--local_batch_size', type=int, default=8, help='the batch size per GPU, local total batch size needs to multiply "local_batch_size" with x(number of GPUs) here we use 6 GPUs, and the global batch size is 48')
    parser.add_argument('--number_of_GPU_used', type=int, default=6, help='number of gpu to use')
    # parser.add_argument('--local_micro_batch_size', type=int, default=32, help='Local micro batch size, 16 for 20news,quail. 32 for GLUE')
    parser.add_argument('--local_num_epochs', type=int, default=2, help='Local number of epochs')
    parser.add_argument('--local_learning_rate', type=float, default=3e-4, help='Local learning rate, 3e-3试过了, for alpaca-lora: 3e-4')

    parser.add_argument('--cutoff_len', type=int, default=512, help='Cutoff length, 512 for GLUE, and 1024 for quail, 2048 for 20news ')
    # the arguments below are for resume training from checkpoint
    parser.add_argument('--resume_from_checkpoint', type=bool, default=False, help='Resume from checkpoint')
    parser.add_argument('--parameter_path', type=str, default='/data/jianhui/checkpoints/roberta-lora/20news-dirichlet_label_uni-0.1-10-NewMethod/aggregated_model_59.bin', help='the parameter path for checkpoint')
    parser.add_argument('--start_round', type=int, default=60, help='the parameter path for checkpoint')
    args = parser.parse_args()
    num_labels_for_each_datasets = {
        'rte':2,
        '20news':20,
    }
    model_paths = {
        # 'roberta': '/home/jianhuiwei/rsch/jianhui/roberta-base',
        'bert': '/home/jianhuiwei/rsch/jianhui/bert-based-uncased',
        'roberta': '/data/jianhui/roberta-base',
        # 'bert': '/data/jianhui/bert-base-uncased',
    }
    output_dirs = {
        'roberta':{
            # 'lora': '/home/jianhuiwei/rsch/jianhui/checkpoints/roberta-lora',
            # 'p_tuningV2': '/home/jianhuiwei/rsch/jianhui/checkpoints/roberta-p_tuningV2',
            # 'IA3': '/home/jianhuiwei/rsch/jianhui/checkpoints/roberta-IA3',
            # 'prefix_tuning': '/home/jianhuiwei/rsch/jianhui/checkpoints/roberta-prefix_tuning',
            'lora': '/data/jianhui/checkpoints/roberta-lora',
            'prefix_tuning': '/data/jianhui/checkpoints/roberta-prefix',
            'IA3': '/data/jianhui/checkpoints/roberta-IA3',
        },
        'bert':{
            'lora': '/home/jianhuiwei/rsch/jianhui/checkpoints/bert-lora',
            'IA3': '/home/jianhuiwei/rsch/jianhui/checkpoints/bert-IA3',
            'p_tuningV2': '/home/jianhuiwei/rsch/jianhui/checkpoints/bert-p_tuningV2',
            'prefix_tuning': '/home/jianhuiwei/rsch/jianhui/checkpoints/bert-prefix_tuning',
        }
    }
    data_paths = {
        # "20news": "/home/jianhuiwei/rsch/jianhui/dataset/20news",
        '20news': "/data/jianhui/dataset/20news"
    }
    test_data_paths = {
        # "20news": "/home/jianhuiwei/rsch/jianhui/dataset/20news/test.json",
        "20news": "/data/jianhui/dataset/20news/test.json",
    }
    args.test_data_path = test_data_paths[args.dataset]
    args.output_dir = output_dirs[args.model][args.peft_method]
    if args.useFedProx:
        args.federated_method='FedProx-' + str(args.proximal_term_argument)
    elif args.useFedAvgM:
        args.federated_method='FedAvgM-' + str(args.beta)
    elif args.useScaffold:
        args.federated_method='Scaffold-' + str(args.local_learning_rate)
    elif args.useHeterogeneityWeight:
        args.federated_method='NewMethod'
    else:
        args.federated_method='FedAvg'
    if args.partition_method == 'iid':
        args.output_dir = os.path.join(args.output_dir, args.dataset +"-"+ args.partition_method + "-" + str(args.num_clients) + "-" + args.federated_method)
    else:
        args.output_dir = os.path.join(args.output_dir, args.dataset +"-"+ args.partition_method + "-"  + str(args.dirichlet_alpha) + "-" + str(args.num_clients) + "-" + args.federated_method)
    if args.partition_method == 'iid':
        args.data_path = os.path.join(data_paths[args.dataset], str(args.num_clients) + "-" + args.partition_method)
    else:
        args.data_path = os.path.join(data_paths[args.dataset], str(args.num_clients) + "-" + args.partition_method +"-"+ str(args.dirichlet_alpha))
    args.num_labels = num_labels_for_each_datasets[args.dataset]
    args.model_path = model_paths[args.model]
    return args