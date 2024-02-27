from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from peft import LoraConfig, get_peft_model
def get_Bert_based_model_and_tokenizer(model_path, num_labels):
    # device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load model to GPU
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model=model.to(device)
    return model, tokenizer

def get_lora_peft_model(model, args):
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        # target_modules=args.lora_target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, config)
    return model, config