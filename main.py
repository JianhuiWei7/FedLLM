import shutil
from utils import (
    setup_seed,
    get_Bert_based_model_and_tokenizer,
    get_lora_peft_model,
    Evaluator,
    evaluate,
)
import time
import datetime
from tqdm import tqdm
import time
import torch
from fed_utils.Scaffold_utils import (
    initialize_server_and_client_control_variate,
    load_variate,
)
from fed_utils.model_aggregation import FedAvg, FedAvgM, ScaffoldAggregation
from fed_utils.client import GenerateClient
from fed_utils.client_participation_scheduling import client_selection
from data_utils import DataTokenizer, partition_data
import os
from peft import (
    set_peft_model_state_dict,
    get_peft_model_state_dict,
)
from parse import parse_args
from accelerate import notebook_launcher
def main(args):
    # to be able to replicate the experiment results
    setup_seed(7)
    # partition data:
    if not os.path.exists(args.data_path):
        partition_data(args=args)
    # build up model and tokenizer
    model, tokenizer = get_Bert_based_model_and_tokenizer(model_path=args.model_path, num_labels=args.num_labels)
    model, config = get_lora_peft_model(model=model, args=args)
    model.print_trainable_parameters()
    
    data_tokenizer = DataTokenizer(args=args, tokenizer=tokenizer)
    if args.useScaffold:
        # initialize server control variate and client control variate.
        dir_name = args.scaffold_dir
        initialize_server_and_client_control_variate(model, args.num_clients, dir_name)
    
    if args.useFedAvgM:
        # initialize server global momentum
        momentum = get_peft_model_state_dict(model=model)

    # if you want to resume training from checkpoint
    # set these parameters
    start_round = 0
    if(args.resume_from_checkpoint):
        peft_weights = torch.load(args.parameter_path)
        set_peft_model_state_dict(model, peft_weights,"default")
        start_round = args.start_round
    
    
    previously_selected_clients_set = set()
    local_dataset_len_dict = dict()
    evaluator = Evaluator(args)
    evaluator.tokenizer = tokenizer
    training_start_time = time.time()
    print("The process of federated instruction-tuning has started..")
    for round in tqdm(range(start_round, args.num_communication_rounds)):
        if args.useScaffold:
            filename = os.path.join(dir_name, "server_c")
            server_c = load_variate(filename)
        else:
            server_c = None
        print("\nConducting the client selection")
        selected_clients_set = client_selection(args.num_clients, args.client_selection_frac, args.client_selection_strategy, other_info=round)
        for client_id in selected_clients_set:
            if args.useScaffold:
                filename = os.path.join(dir_name, "client"+str(client_id))
                client_c = load_variate(filename)
            else:
                client_c = None
            client = GenerateClient(args, client_id, model, args.output_dir, client_c, server_c)
            print("\nPreparing the local dataset and trainer for Client_{}".format(client_id))
            client.load_raw_load()
            client.preprare_local_dataset(data_tokenizer.generate_and_tokenize_prompt)
            
            client.build_local_trainer(tokenizer,
                                       args.local_batch_size,
                                       args.local_num_epochs,
                                       args.local_learning_rate)
            print("Initiating the local training of Client_{}".format(client_id))
            client.initiate_local_training()
            print("Local training starts ... ")
            client.train()
            
            print("\nTerminating the local training of Client_{}".format(client_id))
            model, local_dataset_len_dict, previously_selected_clients_set = client.terminate_local_training(
                round, local_dataset_len_dict, previously_selected_clients_set)
            del client

        print("Collecting the weights of clients and performing aggregation")
        if args.useScaffold:
            model = ScaffoldAggregation(model,
                        selected_clients_set,
                        args.output_dir,
                        local_dataset_len_dict,
                        round,
                        server_c,
                        args.scaffold_dir,
                        args.num_clients,
                        )
        elif args.useFedAvgM:
            model, momentum = FedAvgM(model,
                                selected_clients_set,
                                args.output_dir,
                                local_dataset_len_dict,
                                round,
                                args.beta,
                                momentum,
                                )
        else:
            model = FedAvg(model,
                        selected_clients_set,
                        args.output_dir,
                        local_dataset_len_dict,
                        round,
                        )

        # save checkpoints every 20 rounds
        if (round+1) % 20 == 0:
            torch.save(get_peft_model_state_dict(model), os.path.join(args.output_dir, "aggregated_model_{}.bin".format(round)))
        # delete the clients's weights to save storage space, optional
        shutil.rmtree(os.path.join(args.output_dir, str(round)))
        config.save_pretrained(args.output_dir)
        # if (epoch+1) % 2 == 0:    
        evaluate(round, evaluator, model, args.dataset)
        print("END OF COMMUNICATION: " + str(round))
    training_over_time = time.time()
    training_time = int(round((training_over_time - training_start_time)))
    print("Total training time: " + str(datetime.timedelta(seconds = training_time)))
if __name__ == "__main__":
    args = parse_args()
    main(args)