from peft import (
    set_peft_model_state_dict,  
    get_peft_model_state_dict,
)
import torch
import os
from torch.nn.functional import normalize
from fed_utils.Scaffold_utils import load_variate, write_variate_to_file

def FedAvg(model, selected_clients_set, output_dir, local_dataset_len_dict, epoch):
    weights_array = normalize(
        torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
                     dtype=torch.float32),
        p=1, dim=0)

    for k, client_id in enumerate(selected_clients_set):
        single_output_dir = os.path.join(output_dir, str(epoch), "local_output_{}".format(client_id),
                                         "pytorch_model.bin")
        single_weights = torch.load(single_output_dir)
        if k == 0:
            weighted_single_weights = {key: single_weights[key] * (weights_array[k]) for key in
                                       single_weights.keys()}
        else:
            weighted_single_weights = {key: weighted_single_weights[key] + single_weights[key] * (weights_array[k])
                                       for key in
                                       single_weights.keys()}

    set_peft_model_state_dict(model, weighted_single_weights, "default")

    return model

def FedAvgM(model, selected_clients_set, output_dir, local_dataset_len_dict, epoch, beta, momentum):
    old_param = get_peft_model_state_dict(model=model)
    weights_array = normalize(
        torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
                     dtype=torch.float32),
        p=1, dim=0)

    for k, client_id in enumerate(selected_clients_set):
        single_output_dir = os.path.join(output_dir, str(epoch), "local_output_{}".format(client_id),
                                         "pytorch_model.bin")
        single_weights = torch.load(single_output_dir)
        if k == 0:
            weights_difference = {key: (old_param[key]-single_weights[key]) * (weights_array[k]) for key in
                                       single_weights.keys()}
        else:
            weights_difference = {key: weights_difference[key] + (old_param[key]-single_weights[key]) * (weights_array[k])
                                       for key in
                                       single_weights.keys()}
    
    new_momentum = {key: momentum[key] * beta + weights_difference[key] for key in momentum.keys()}
    new_param = {key: old_param[key] - new_momentum[key] for key in old_param.keys()}
    set_peft_model_state_dict(model, new_param, "default")
    return model, new_momentum

def ScaffoldAggregation(model, selected_clients_set, output_dir, local_dataset_len_dict, epoch, server_c, dir_name, num_clients):
    weights_array = normalize(
        torch.tensor([local_dataset_len_dict[client_id] for client_id in selected_clients_set],
                     dtype=torch.float32),
        p=1, dim=0)

    for k, client_id in enumerate(selected_clients_set):
        single_output_dir = os.path.join(output_dir, str(epoch), "local_output_{}".format(client_id),
                                         "pytorch_model.bin")
        single_weights = torch.load(single_output_dir)
        if k == 0:
            weighted_single_weights = {key: single_weights[key] * (weights_array[k]) for key in
                                       single_weights.keys()}
        else:
            weighted_single_weights = {key: weighted_single_weights[key] + single_weights[key] * (weights_array[k])
                                       for key in
                                       single_weights.keys()}

    set_peft_model_state_dict(model, weighted_single_weights, "default")
    
    server_c = {}
    for index, i in enumerate(selected_clients_set):
        filename = os.path.join(dir_name, "client"+str(i))
        local_variate = load_variate(filename=filename)
        for k,v in local_variate.items():
            if index == 0:
                server_c[k] = v.data / num_clients
            else:
                server_c[k] += v.data / num_clients
    filename = os.path.join(dir_name, "server_c")
    write_variate_to_file(filename=filename, variate=server_c)

    return model