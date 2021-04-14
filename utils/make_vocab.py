import os
import pandas as pd
import numpy as np


def make_vocab(vocab_path):
    if os.path.exists(vocab_path):
        return

    with open(vocab_path, 'w') as t:
        for i in range(3300):
            t.write(str(i) + '\n')

        for s in ["[UNK]", "[SEP]", "[CLS]", "[MASK]", "[PAD]"]:
            t.write(s + '\n')


def normalize(csv):
    count = 0

    for col_name, item in csv.iteritems():
        item_min = min(item)
        item_max = max(item)

        if item_min != item_max:
            item = round((item - item_min) / item_max * 9)

        item = item + 10 * count
        count += 1

        csv[col_name] = item

    return csv


def write_string(save_path, s):
    with open(save_path, "a+") as f:
        f.write(s)


def csv_to_string(csv_path, save_path):
    csv = pd.read_csv(csv_path, encoding="cp949")
    csv = normalize(csv)

    data = csv.to_numpy().astype(np.int32).astype(str)

    for d in data:
        s = ""

        for i in d:
            s = s + i + " "
        s += '\n'

        write_string(save_path, s)