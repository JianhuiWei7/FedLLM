
class DataTokenizer:
    def __init__(self, args, tokenizer) -> None:
        self.args = args
        self.tokenizer = tokenizer
    def generate_and_tokenize_prompt(self, data_point):
        return self.generate_and_tokenize_for_bert_based(data_point)

    def generate_and_tokenize_for_bert_based(self, data_ponit):
        # add_special_tokens: add <s> as CLS token and </s> as SEP token
        # if self.args.peft_method == 'prefix_tuning':
        #     cut_off_length = self.args.cutoff_len - self.args.num_virtual_tokens
        # else:
        #     cut_off_length = self.args.cutoff_len
        result = self.tokenizer(text=data_ponit['context'], padding='max_length', add_special_tokens=True,truncation=True, max_length=self.args.cutoff_len)
        result['label'] = data_ponit['id_of_label']
        return result