import json
from datasets import Dataset
from sklearn.datasets import fetch_20newsgroups


def main():
    
    for split in ["train", "test"]:
        # Follow recommendation to strip newsgroup metadata
        # https://scikit-learn.org/0.19/datasets/twenty_newsgroups.html#filtering-text-for-more-realistic-training
        data = fetch_20newsgroups(subset=split, remove=("headers", "footers", "quotes"))
        id2label = {idx: label for idx, label in enumerate(data["target_names"])}
        d = {"text": data["data"], "label": data["target"]}
        dset = Dataset.from_dict(d)
        dset = dset.map(lambda x: {"label_text": id2label[x["label"]]})
        dset.to_json(f"{split}.jsonl")

def generate_original_data_as_json():
    train_data = []
    test_data = []
    split = "train"
    data = fetch_20newsgroups(subset=split, remove=("headers", "footers", "quotes"))
    id2label = {idx: label for idx, label in enumerate(data["target_names"])}
    for text, target in zip(data['data'], data['target']):
        instance = {}
        instance['text'] = text
        instance['target'] = str(target)
        instance['category'] = id2label[target]
        train_data.append(instance)
    with open("/home/jianhuiwei/rsch/jianhui/dataset/20news/original_train.json", 'w') as write_f:
        write_f.write(json.dumps(train_data, indent=4))
    split = "test"
    data = fetch_20newsgroups(subset=split, remove=("headers", "footers", "quotes"))
    id2label = {idx: label for idx, label in enumerate(data["target_names"])}
    for text, target in zip(data['data'], data['target']):
        instance = {}
        instance['text'] = text
        instance['target'] = str(target)
        instance['category'] = id2label[target]
        test_data.append(instance)
    with open("/home/jianhuiwei/rsch/jianhui/dataset/20news/original_test.json", 'w') as write_f:
        write_f.write(json.dumps(test_data, indent=4))

def generate_train_from_original_json():
    data_file = open("/home/jianhuiwei/rsch/jianhui/dataset/20news/original_train.json", 'r')
    content = data_file.read()
    original_data = json.loads(content)
    generated_data = []
    for item in original_data:
        if len(item['text']) >= 512:
            item['text'] = item['text'][0:512]
        instance = {}
        instance['instruction'] = "A piece of short news report will be presented to you, please categorize it. "
        # long_instruction = "options: \
        # 0: alt.atheism, \
        # 1:comp.graphics, \
        # 2:comp.os.ms-windows.misc, \
        # 3:comp.sys.ibm.pc.hardware, \
        # 4:comp.sys.mac.hardware, \
        # 5:comp.windows.x, \
        # 6:misc.forsale, \
        # 7:rec.autos, \
        # 8:rec.motorcycles, \
        # 9:rec.sport.baseball, \
        # 10:rec.sport.hockey, \
        # 11:sci.crypt, \
        # 12:sci.electronics, \
        # 13:sci.med, \
        # 14:sci.space, \
        # 15:soc.religion.christian, \
        # 16:talk.politics.guns, \
        # 17:talk.politics.mideast, \
        # 18:talk.politics.misc, \
        # 19:talk.religion.misc."
        instance['context'] = item['text']
        instance['response'] = item['target']
        instance['id_of_label'] = int(item['target'])
        instance['category'] = item['category']
        generated_data.append(instance)
    with open("/home/jianhuiwei/rsch/jianhui/dataset/20news/train.json", 'w') as write_f:
        write_f.write(json.dumps(generated_data, indent=4))
def generate_test_from_original_json():
    data_file = open("/home/jianhuiwei/rsch/jianhui/dataset/20news/original_test.json", 'r')
    content = data_file.read()
    original_data = json.loads(content)
    generated_data = []
    maxLen = 100
    for item in original_data:
        if len(item['text']) >= 512:
            item['text'] = item['text'][0:512]
        if len(item['text']) > maxLen:
            maxLen = len(item['text'])
        instance = {}
        instance['instruction'] = "A piece of short news report will be presented to you, please categorize it."
        instance['context'] = item['text']
        instance['response'] = item['target']
        instance['category'] = item['category']
        instance['id_of_label'] = int(item['target'])
        generated_data.append(instance)
    print(maxLen)
    with open("/home/jianhuiwei/rsch/jianhui/dataset/20news/test.json", 'w') as write_f:
        write_f.write(json.dumps(generated_data, indent=4))
if __name__ == "__main__":
    # generate_original_data_as_json()
    generate_train_from_original_json()
    generate_test_from_original_json()