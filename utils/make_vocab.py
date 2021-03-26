import os

def make_vocab(vocab_path):
    if os.path.exists(vocab_path):
        return

    with open(vocab_path, 'w') as t:
        for i in range(3300):
            t.write(str(i) + '\n')

        for s in ["[UNK]", "[SEP]", "[CLS]", "[MASK]", "[PAD]"]:
            t.write(s + '\n')


if __name__ == "__main__":
    make_vocab("vocab.txt")