__author__ = "Ji-Hwan Moon"

import argparse

import utils.utils as util
import model.train as train
from utils.data_provider_for_GPT import DataProvider as DP

from transformers import RobertaTokenizerFast
from transformers import pipeline
from transformers import RobertaForMaskedLM


class MedicalAnalysis:
    def __init__(self):
        self.args = self.get_args()

        self.tokenizer = self.get_tokenizer()
        self.model = RobertaForMaskedLM.from_pretrained(self.args.model_path)
        self.fill_mask = pipeline("fill-mask",
                                  model=self.args.model_path,
                                  tokenizer=self.tokenizer)

        self.last_attentions = []

    def make_string(self, L):
        ret = ""
        L = L[:-7]

        for l in L:
            ret = ret + l + " "

        return ret

    def get_tokenizer(self):
        # load tokenizer from pretrained tokenizer
        tokenizer = RobertaTokenizerFast.from_pretrained(self.args.tokenizer_path, max_len=512)

        return tokenizer

    def start(self):
        with open("csv_string.txt") as cs:
            L = cs.readlines()
            PD = []
            PB = []

            count = 0
            for l in L:
                l2 = list(l.split())
                s = self.make_string(l2)

                prediction, probability = self.get_answer(s)
                PD.append(prediction)
                PB.append(probability)
                count += 1

            return PD, PB, count

    def get_answer(self, s):
        probability = []
        for _ in range(7):
            s += "<mask>"
            s_model = self.tokenizer(s, return_tensors="pt")

            self.last_attentions = self.model(**s_model, output_attentions=True)
            self.last_attentions = self.last_attentions[-1]

            output_pipe = self.fill_mask(s)

            s = output_pipe[0]["sequence"]
            probability.append(output_pipe[0]["score"])

        average_pro = sum(probability) / len(probability)
        return s, average_pro

    def get_args(self):
        parser = argparse.ArgumentParser(description="For medical information analysis")

        parser.add_argument('--data_path', required=False, default='data.csv')
        parser.add_argument('--tokenizer_path', required=False, default='model/tokenizer_model/')
        parser.add_argument('--model_path', default='model/transformer_model/')
        parser.add_argument('--is_train_model', required=False, default=False)
        parser.add_argument('--is_train_tokenizer', default=False)
        parser.add_argument('--csv_string_path', default="csv_string.txt")

        args = parser.parse_args()

        return args


txt_dir = "model/train.txt"
tokenizer_dir = "model/tokenizer_model/"
model_dir = "model/transformer_model/"
train.model_sequence(txt_dir, tokenizer_dir, model_dir, False, True)

# if __name__ == "__main__":
#     dp = DP("utils/train.csv")
#     util.make_train_txt("utils/test.csv", "model/test.txt")
#     m = MedicalAnalysis()
#
#     PD, PB, count = m.start()
#
#     param = dp.param.to_numpy()[-7:]
#
#     denorm = dp.denormalize(PD)
#
#     for i in range(len(PD)):
#         print(f"{i} test result \nprediction : {PB[i]}")
#
#         for j in range(7):
#             print(f"{param[j]} : {denorm[i][j][0]} ~ {denorm[i][j][1]}")
#
#         print()