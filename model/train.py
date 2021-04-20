from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

paths = "string_csv.txt"

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer()

# Customize training
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

save_path_tok = "/content/gdrive/MyDrive/GPT2/tokenizer/"
tokenizer.save_model(save_path_tok)

tokenizer = ByteLevelBPETokenizer(
    save_path_tok + "vocab.json",
    save_path_tok + "merges.txt"
)