from utils.data_provider_for_GPT import DataProvider as DP
from model import train
from utils import utils

from transformers import RobertaTokenizerFast
from transformers import pipeline
from transformers import RobertaForMaskedLM

import random
import pandas as pd
from tqdm import tqdm


def make_string(L):
    ret = ""
    L = L[:-7]

    for l in L:
        ret = ret + l + " "

    return ret


def check_position(output_pipe, i):
    token = str(320 + i)
    d = output_pipe[0]

    score = d["score"]

    token_str = d["token_str"]
    len_token_str = len(token_str) - 1

    if len_token_str != 0:
        seq = d["sequence"][:-len_token_str]
        seq = seq + token + token_str[-1]
        mask = token + token_str[-1]
    else:
        seq = d["sequence"] + token + "0"
        mask = token + "0"

    return seq, score, mask


def get_answer(fill_mask, s):
    s = s[:-1]
    probability = []
    masks = []
    for i in range(7):
        s += " <mask>"
        # s_model = self.tokenizer(s, return_tensors="pt")

        # self.last_attentions = self.model(**s_model, output_attentions=True)
        # self.last_attentions = self.last_attentions[-1]

        output_pipe = fill_mask(s)

        seq, score, mask = check_position(output_pipe, i)
        s = seq
        probability.append(score)
        masks.append(mask)

    average_pro = sum(probability) / len(probability)
    return s, average_pro, masks


def start(fill_mask):
    with open("model/test.txt") as cs:
        L = cs.readlines()
        PD = []
        PB = []
        ill = []

        count = 0
        for l in L:
            l2 = list(l.split())
            s = make_string(l2)

            prediction, probability, illness = get_answer(fill_mask, s)
            PD.append(prediction)
            PB.append(probability) # 학습 확률
            ill.append(illness)
            count += 1

        return PD, PB, ill, count

# test 모듈
def test_sequence():
    result_dir = "../results/"

    tok_dir = "tokenizer_model/"
    tokenizer = train.get_tok(tok_dir)

    csv_path = "../utils/test.csv"
    txt_name = "test.txt"
    utils.make_train_txt(csv_path, txt_name)

    dp = DP("../utils/test.csv")
    param = dp.param.to_numpy()[-7:]
    param = param.tolist()
    param.append("probability")

    mod_dir = "transformer_model/checkpoint-33000/"
    model = RobertaForMaskedLM.from_pretrained(mod_dir + str(i))
    fill_mask = pipeline("fill-mask",
                         model=model,
                         tokenizer=tokenizer)

    PD, PB, ill, count = start(fill_mask)

    denorm = dp.denormalize(ill)

    l = []
    for x in range(len(PD)):
        tmp = []

        for j in range(7):
            tmp.append(random.randrange(denorm[x][j][0], denorm[x][j][1] + 1))
        tmp.append(PB[x])

        l.append(tmp)

    df = pd.DataFrame(l, columns=param)
    df.to_csv(result_dir + "result.csv", index=False, encoding="cp949")

test_sequence()