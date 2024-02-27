# -*- coding:utf-8 -*-
"""
@Time: 2022/03/02 13:34
@Author: KI
@File: ScaffoldOptimizer.py
@Motto: Hungry And Humble
"""
import math
from transformers import Trainer
from torch.optim import Optimizer
import pickle
import os
import torch
from torch import nn
class ScaffoldOptimizer(Optimizer):
    def __init__(self, params, lr, server_c, client_c):
        defaults = dict(lr=lr, server_c=server_c, client_c=client_c)
        # self.server_c = server_c
        # self.client_c = client_c
        super(ScaffoldOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        # 
        loss = None
        if closure is not None:
            loss = closure()
        
        tunable_parameters = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                else:
                    tunable_parameters.append(p)
            for p, c, ci in zip(tunable_parameters, group['server_c'].values(), group['client_c'].values()):
                c = c.to(p.device)
                ci = ci.to(p.device)
                dp = p.grad.data + c.data - ci.data
                p.data = p.data - dp.data * group['lr']

        return loss
        """
        below is the code for Adam or AdamW Optimizer.
        """
        # """Performs a single optimization step.

        #     Arguments:
        #         closure (callable, optional): A closure that reevaluates the model
        #             and returns the loss.
        #     """
        #     loss = None
        #     if closure is not None:
        #         loss = closure()

        #     for group in self.param_groups:
        #         for p in group['params']:
        #             if p.grad is None:
        #                 continue
        #             grad = p.grad.data
        #             if grad.is_sparse:
        #                 raise RuntimeError('Adam does not support sparse gradients, '
        #                                     'please consider SparseAdam instead')
        #             amsgrad = group['amsgrad']

        #             state = self.state[p]

        #             # State initialization
        #             if len(state) == 0:
        #                 state['step'] = 0
        #                 # Exponential moving average of gradient values
        #                 state['exp_avg'] = torch.zeros_like(p.data)
        #                 # Exponential moving average of squared gradient values
        #                 state['exp_avg_sq'] = torch.zeros_like(p.data)
        #                 if amsgrad:
        #                     # Maintains max of all exp. moving avg. of sq. grad. values
        #                     state['max_exp_avg_sq'] = torch.zeros_like(p.data)

        #             exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        #             if amsgrad:
        #                 max_exp_avg_sq = state['max_exp_avg_sq']
        #             beta1, beta2 = group['betas']

        #             state['step'] += 1
        #             bias_correction1 = 1 - beta1 ** state['step']
        #             bias_correction2 = 1 - beta2 ** state['step']
                    
        #             # if the optimiezer is Adam, L2 regularization is added here.
        #             # if group['weight_decay'] != 0:
        #             #     grad.add_(group['weight_decay'], p.data)

        #             # Decay the first and second moment running average coefficient
        #             exp_avg.mul_(beta1).add_(1 - beta1, grad)
        #             exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        #             if amsgrad:
        #                 # Maintains the maximum of all 2nd moment running avg. till now
        #                 torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
        #                 # Use the max. for normalizing running avg. of gradient
        #                 denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
        #             else:
        #                 denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

        #             step_size = group['lr'] / bias_correction1

        #             p.data.addcdiv_(-step_size, exp_avg, denom)
                    
        #             # if use AdamW, weight decay is used here.
        #             if group["weight_decay"] > 0.0:
        #                 p.data.add_(p.data, alpha=(-group["lr"] * group["weight_decay"]))
        #     return loss

def write_variate_to_file(variate, filename):
    with open(filename, 'wb') as write_f:
        pickle.dump(variate, write_f)                     
    return filename

def load_variate(filename):
    with open(filename, 'rb') as f:
        variate = pickle.load(f)
    return variate
    
def initialize_server_and_client_control_variate(model, num_clients, dir_name):
    os.makedirs(name = dir_name, exist_ok=True)
    control_variate  = {}
    total_num_of_parameters = 0
    num_parameters_for_peft = 0
    # model = get_peft_model_state_dict(model)
    for k, v in model.named_parameters():
        if v.requires_grad == False:
            total_num_of_parameters+=1
            continue
        else:
            total_num_of_parameters+=1
            num_parameters_for_peft+=1
            control_variate[k] = torch.zeros_like(v.data)
    filename = os.path.join(dir_name, 'server_c')
    write_variate_to_file(control_variate, filename)
    for i in range(num_clients):
        filename = os.path.join(dir_name, 'client' + str(i))
        write_variate_to_file(control_variate, filename)

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(20, 20)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(20, 40)
        self.fc3 = nn.Linear(40, 20)
        self.fc4 = nn.Linear(20, 1)

    def forward(self, data):
        x = self.fc1(data)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        x = self.fc4(x)
        x = self.sigmoid(x)

        return x
if __name__ == "__main__":
    ann = ANN()
    dir_name = './scaffold_control_variate'
    initialize_server_and_client_control_variate(ann, 2, dir_name)
    server_c = load_variate('./scaffold_control_variate/server_c')
    client_c = load_variate('./scaffold_control_variate/client0')
    optimizer1 = ScaffoldOptimizer(ann.parameters(), lr=0.001, server_c=server_c, client_c=client_c)
    optimizer1.step()