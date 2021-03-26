import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer


def get_token_param(data):
    return data[:, :-7].tolist(), data[:, -7:].tolist()


class FoodDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)


class DataProvider:
    def __init__(self, path, vocab_path):
        self.path = path
        self.csv = pd.read_csv(self.path, encoding="cp949")

        self.normalize()

        self.param = self.csv.keys()
        self.data = self.csv.to_numpy().astype(np.int32).astype(str)

        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.train_labels = None
        self.val_labels = None
        self.test_labels = None

        self.split_data()

        self.tokenizer = BertTokenizer(vocab_file=vocab_path, tokenize_chinese_chars=False)
        self.train_encoding = self.tokenizer(*get_token_param(self.train_data), return_tensors="pt")
        self.val_encoding = self.tokenizer(*get_token_param(self.val_data), return_tensors="pt")
        self.test_encoding = self.tokenizer(self.test_data[:, :-7].tolist(),
                                            self.test_data[:, -7:].tolist(),
                                            return_tensors="pt")

        self.train_dataset = FoodDataset(self.train_encoding, self.train_labels)
        self.vat_dataset = FoodDataset(self.val_encoding, self.val_labels)
        self.test_dataset = FoodDataset(self.test_encoding, self.test_labels)

    def split_data(self):
        self.train_data, self.val_data = train_test_split(self.data[:-100], test_size=.2)
        self.train_labels, self.val_labels = train_test_split(np.array(['1'] * (self.data.shape[0] - 100)), test_size=.2)
        self.test_data, self.test_labels = self.data[-100:], np.array(['1'] * 100)

    def normalize(self):
        count = 0

        for col_name, item in self.csv.iteritems():
            item_min = min(item)
            item_max = max(item)

            if item_min != item_max:
                item = round((item - item_min) / item_max * 9)

            item = item + 10 * count
            count += 1

            self.csv[col_name] = item




if __name__ == "__main__":
    data_path = '../data.csv'
    dp = DataProvider(data_path)