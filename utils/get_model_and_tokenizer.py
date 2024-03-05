from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, PrefixTuningConfig, IA3Config
import torch
def get_Bert_based_model_and_tokenizer(args):
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load model to GPU
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=args.model_path, num_labels=args.num_labels)
    if args.peft_method == 'p_tuningV2' and args.model == 'bert':
        max_length = args.cutoff_len - (args.num_virtual_tokens+2)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, model_max_length=max_length)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model=model.to(device)
    return model, tokenizer

def return_peft_model(model, args):
    if args.peft_method == 'lora':
        return get_lora_peft_model(model=model, args=args)
    elif args.peft_method == 'p_tuningV2':
        return get_p_tuningV2_peft_model(model=model, args=args)
    elif args.peft_method == 'IA3':
        return get_IA3_peft_model(model=model, args=args)
    elif args.peft_method == "prefix_tuning":
        return get_prefix_tuning_peft_model(model=model, args=args)

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

def get_p_tuningV2_peft_model(model, args):
    # since we didn't employ MLP in virtual tokens, so it is P-tuning V2
    config = PrefixTuningConfig(
        task_type="SEQ_CLS",
        num_virtual_tokens=args.num_virtual_tokens,
    )
    model = get_peft_model(model, config)
    return model, config

def get_prefix_tuning_peft_model(model, args):
    config = PrefixTuningConfig(
        task_type="SEQ_CLS",
        num_virtual_tokens=args.num_virtual_tokens,
        prefix_projection=True,
    )
    model = get_peft_model(model, config)
    return model, config

def get_IA3_peft_model(model, args):
    config = IA3Config(
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, config)
    return model, config

# def get_BitFit_peft_model(modedl, args):
#     for k, v in modedl.named_parameters():
