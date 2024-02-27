from fed_utils.FedProx_utils import FedProxTrainer
import transformers

def select_trainer(useFedProx):
    if useFedProx:
        return FedProxTrainer
    else:
        return transformers.Trainer
