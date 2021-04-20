from utils.data_provider_for_GPT import DataProvider as DP

def change_to_string(num_list):
    ret = ""

    for n in num_list:
        ret += (str(n) + " ")

    return ret

def csv_to_token(path):
    dp = DP(path)
    data = dp.data
    tokens = []

    for d in data:
        t = change_to_string(d)
        tokens.append(t)

    return tokens

def make_train_txt(path, name):
    tokens = csv_to_token(path)

    with open(name, 'w') as txt:
        for t in tokens:
            txt.write(t + '\n')

# def make_test_set(start, tmp, save_path):
#     if start == 3270:
#         with open(save_path, 'a+') as t:
#             t.write(change_to_string(tmp) + '\n')
#         return
#
#     for i in range(10):
#         num = start + i
#         tmp.append(num)
#         make_test_set(start + 10, tmp, save_path)
#         tmp.pop()

#
# if __name__ == "__main__":
#     make_test_set(3200, [], "test_sample.txt")