import os
import re
import sys
sys.path.append("./")
import torch
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass
class Evaluator():
    def __init__(self, args):
        self.args = args
        self.tokenizer = None
        self.model = None
        self.testset_path =  {
            "20news": "/home/jianhuiwei/rsch/jianhui/dataset/20news/test.json",
        }
        self.save_path = {
            "20news": "./output/20news",
        }
        self.output_directory = self.save_path[args.dataset]
        if args.partition_method == 'iid':
            self.output_directory = self.output_directory + "-" + args.model + "-" + args.peft_method + "-" + args.partition_method + "-" + str(args.num_clients) + "-" + args.federated_method
        else:
            self.output_directory = self.output_directory + "-" + args.model + "-" + args.peft_method + "-" + args.partition_method + "-" + str(args.dirichlet_alpha) + "-" + str(args.num_clients) + "-" + args.federated_method
        
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
        self.output_short_result_file_name = os.path.join(self.output_directory, "short_result.txt")
        testset = load_dataset("json", data_files=self.testset_path[args.dataset])
        cols = ['instruction', 'response', 'context', 'category']
        cleared_testset = testset["train"].shuffle().map(self.generate_prompt_for_bert_based, remove_columns=cols)
        cleared_testset.set_format(type="torch", columns=["full_prompt", "label"])
        self.dataloader = DataLoader(cleared_testset, batch_size=args.local_batch_size, drop_last=False)

    
    def batch_run(self, batch_input):
        tokenized_inputs = self.tokenizer(batch_input['full_prompt'], padding='max_length', max_length=self.args.cutoff_len, return_tensors="pt")
        tokenized_inputs = tokenized_inputs.to(device)
        outputs = self.model(**tokenized_inputs)
        logits = outputs.logits
        response = torch.argmax(logits, dim=-1)
        response = [res.cpu() for res in response]
        return response
    
    def generate_prompt_for_bert_based(self, data_point):
        data_dict = {
            "full_prompt": data_point['context'],
            "label": data_point['id_of_label']
        }
        return data_dict
    
    def write_to_file(self, index, result):
        with open(self.output_short_result_file_name, 'a') as f:
            f.write(str(index) + " " + str(result) + '\n')

def evaluate(rnd, evaluator, model, dataset):
    model.eval()
    evaluator.model = model
    all = 0
    correct = 0
    with torch.no_grad():
        for batch in tqdm(evaluator.dataloader, desc="Evaluating"):
            list_of_response = evaluator.batch_run(batch)
            for pred, label in zip(list_of_response, batch['label']):
                if(pred == label):
                    correct += 1
            all += len(batch['label'])
            acc = correct / all
            print(f"Accuracy of the {dataset} dataset: {acc:.4f} (Correct: {correct}, Total: {all})")
    short_result = str(acc)
    evaluator.write_to_file(index=rnd, result=short_result)


