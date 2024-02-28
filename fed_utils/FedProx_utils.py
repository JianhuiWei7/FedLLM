import math
import torch
from transformers import Trainer
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch.optim import Optimizer
from collections import OrderedDict
def compute_proximal_term(original_peft_weights, updated_model, device):
    # define how to compute difference loss
    proximal_term = 0.0
    new_peft_weights = OrderedDict((name, param) for name, param in updated_model.named_parameters() if "default" in name)
    for original_weights, updated_weights in zip(original_peft_weights.values(), new_peft_weights.values()):
        original_weights = original_weights.to(device)
        updated_weights = updated_weights.to(device)
        proximal_term += torch.norm((original_weights - updated_weights), p=2)
    
    return proximal_term

    
class FedProxTrainer(Trainer):
    def set_previous_peft_weights(self, peft_weights):
        self.previous_peft_weights = peft_weights

    def set_proximal_term_mu(self, arg):
        self.proximal_term_mu = arg
        
    def compute_loss(self, model, inputs, return_outputs=False):
        (loss, outputs) = super().compute_loss(model, inputs, return_outputs=True)
        loss_device = loss.device
        proximal_term = compute_proximal_term(original_peft_weights=self.previous_peft_weights, updated_model=model, device=loss_device)
        proximal_term = proximal_term.to(loss_device)
        # print("proximal term loss: " + str(0.5 * self.proximal_term_mu * proximal_term))
        # print("loss: ", loss)
        loss += 0.5 * self.proximal_term_mu * proximal_term
        # print("proximal_term_loss: ", 0.5 * self.proximal_term_mu * proximal_term)
        return (loss, outputs) if return_outputs else loss
    
class FedProx(Optimizer):

    def __init__(self, params, mu, lr=1e-3,betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        self.mu = mu
        defaults = dict(lr=lr, betas=betas, eps=eps,weight_decay=weight_decay, amsgrad=amsgrad)
        super(FedProx, self).__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, '
                                        'please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0 
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                # old_init只会被初始化一次。
                if 'old_init' not in state:
                    state['old_init'] = torch.clone(p.data).detach()
                grad.add_(self.mu, p.data - state['old_init'])
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # if the optimiezer is Adam, L2 regularization is added here.
                # if group['weight_decay'] != 0:
                #     grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)
                
                # if use AdamW, weight decay is used here.
                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=(-group["lr"] * group["weight_decay"]))
        return loss
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                
                param_state = self.state[p]
                # old_init只会被初始化一次。
                if 'old_init' not in param_state:
                    param_state['old_init'] = torch.clone(p.data).detach()

                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                # apply proximal update
                # 已经被求导过了，所以不需要1/2 || ^2那样的写法
                # 直接在一个参数的grad上面加上新的grad。
                d_p.add_(self.mu, p.data - param_state['old_init'])
                p.data.add_(-group['lr'], d_p)

        return loss
