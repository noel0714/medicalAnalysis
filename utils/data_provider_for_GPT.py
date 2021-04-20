import pandas as pd
import numpy as np


class DataProvider:
    def __init__(self, path):
        self.path = path
        self.csv = pd.read_csv(self.path, encoding="cp949")
        self.minmax = []

        self.get_minmax()
        self.normalize()

        self.param = self.csv.keys()
        self.data = self.csv.to_numpy().astype(np.int32).astype(str)

    def get_minmax(self):
        path = "utils/max_value.txt"

        with open(path) as t:
            l = t.readline()
            self.minmax = list(map(float, l.split()))
            self.minmax = list(map(int, self.minmax))

    def normalize(self):
        count = 0

        for col_name, item in self.csv.iteritems():
            item_min = 0
            item_max = self.minmax[count]

            if item_min != item_max:
                item = round((item - item_min) / item_max * 9)

            item = item + 10 * count
            count += 1

            self.csv[col_name] = item

    def denormalize(self, prediction):
        ret = []
        pred = list(s.split() for s in prediction)
        pred = list(l[-7:] for l in pred)

        for p in pred:
            tmp = []
            for i in p:
              this_min, this_max = self.minmax[int(i[:-1])]
              this_rank = int(i[-1])

              low_bound = this_rank / 9 * this_max + this_min
              high_bound = (this_rank + 1) / 9 * this_max + this_min

              tmp.append([round(low_bound), round(high_bound)])

            ret.append(tmp)
        return ret


if __name__ == "__main__":
    data_path = '../data.csv'
    dp = DataProvider(data_path)

