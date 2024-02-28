import math
import transformers
import os
from datasets import load_dataset
import copy
from collections import OrderedDict
import torch
from fed_utils.FedProx_utils import FedProxTrainer
from peft import (
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from torch.utils.data import DataLoader
from fed_utils.Scaffold_utils import ScaffoldOptimizer, write_variate_to_file
from fed_utils.FedProx_utils import FedProx
from utils.select_trainer_optimizer import select_trainer
class GenerateClient:
    def __init__(self, args, client_id, model, output_dir, client_c=None, server_c=None):
        self.args = args
        self.client_id = client_id
        self.model = model
        self.output_dir = output_dir
        self.local_output_dir = os.path.join(self.output_dir, "trainer_saved", "local_output_{}".format(self.client_id))
        self.data_path = args.data_path
        self.client_c = client_c
        self.server_c = server_c
    def load_raw_load(self):
        self.local_data_path = os.path.join(self.data_path, "local_training_{}.json".format(self.client_id))
        self.local_data = load_dataset("json", data_files=self.local_data_path)
        
    def preprare_local_dataset(self, generate_and_tokenize_prompt):
        cols = ['instruction', 'response', 'context', 'category'] 
        self.local_train_dataset = self.local_data["train"].shuffle().map(generate_and_tokenize_prompt, remove_columns=cols)
        self.local_eval_dataset = None

    def build_local_trainer(self,
                            tokenizer,
                            local_batch_size,
                            local_num_epochs,
                            local_learning_rate,
                            ):
        self.train_args = transformers.TrainingArguments(
            per_device_train_batch_size=local_batch_size,
            num_train_epochs=local_num_epochs,
            learning_rate=local_learning_rate,
            logging_strategy="steps",
            logging_steps=1,
            output_dir=self.local_output_dir,
            dataloader_drop_last=False,
            ddp_find_unused_parameters=False
        )
        # select optimizer
        if self.args.useScaffold:
            optimizer = ScaffoldOptimizer(
                params = self.model.parameters(),
                lr = local_learning_rate,
                server_c = self.server_c,
                client_c = self.client_c
            )
        else:
            # use the default optimizer
            optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=local_learning_rate)
        # setup local lr_scheduler
        dataset_len = len(self.local_train_dataset)
        steps = math.ceil(dataset_len / (self.args.local_batch_size)) * self.args.local_num_epochs
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1, end_factor=0, total_iters=steps)
        # select data_collator
        data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
        # select trainer
        trainer = select_trainer(self.args.useFedProx)
        self.local_trainer = trainer(
                model=self.model,
                optimizers=(optimizer, lr_scheduler),
                train_dataset=self.local_train_dataset,
                eval_dataset=self.local_eval_dataset,
                args=self.train_args,
                data_collator=data_collator
            )

    def initiate_local_training(self):
        self.model.config.use_cache = False
        self.params_dict_old = copy.deepcopy(get_peft_model_state_dict(self.model))

    def train(self):
        if self.args.useFedProx:
            self.local_trainer.set_previous_peft_weights(self.params_dict_old)
            self.local_trainer.set_proximal_term_mu(self.args.proximal_term_argument)
        self.local_trainer.train()

    # def train_trainer_ddp(self,
    #                       tokenizer,
    #                       local_micro_batch_size,
    #                       local_num_epochs,
    #                       local_learning_rate,
    #                       ):
    #     self.build_local_trainer(tokenizer,local_micro_batch_size,local_num_epochs,local_learning_rate)
    #     self.initiate_local_training()
    #     self.train()

    def terminate_local_training(self, epoch, local_dataset_len_dict, previously_selected_clients_set):
        # update local control variate and save it to file
        
        if self.args.useScaffold:
            local_steps = (len(self.local_train_dataset) // self.args.local_batch_size)*self.args.local_num_epochs
            for k, v in self.model.named_parameters():
                if v.requires_grad == True:
                    # change the name of the weight
                    # ['base_model.model.classifier.dense.weight', 'base_model.model.classifier.dense.bias', 'base_model.model.classifier.out_proj.weight', 'base_model.model.classifier.out_proj.bias']
                    if 'classifier' in k:
                        ki = k.replace('.modules_to_save.default', '')
                    else:
                        ki = k[0:-14] + 'weight'
                    self.client_c[k] = self.client_c[k] - self.server_c[k] + (self.params_dict_old[ki].data - v.data) / (self.args.local_learning_rate*local_steps)
            filename = os.path.join(self.args.scaffold_dir, "client"+str(self.client_id))
            write_variate_to_file(filename=filename, variate=self.client_c)
        
        local_dataset_len_dict[self.client_id] = len(self.local_train_dataset)
        new_adapter_weight = get_peft_model_state_dict(self.model)
        
        
        single_output_dir = os.path.join(self.output_dir, str(epoch), "local_output_{}".format(self.client_id))
        os.makedirs(single_output_dir, exist_ok=True)
        torch.save(new_adapter_weight, single_output_dir + "/pytorch_model.bin")

        set_peft_model_state_dict(self.model, self.params_dict_old, "default")
        previously_selected_clients_set = previously_selected_clients_set | set({self.client_id})

        return self.model, local_dataset_len_dict, previously_selected_clients_set