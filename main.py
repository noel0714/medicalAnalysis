__author__ = "Ji-Hwan Moon"

import argparse
import torch
import datetime

from torch.utils.data import DataLoader
from utils import data_provider, make_vocab
from transformers import BertForNextSentencePrediction, Trainer, TrainingArguments, BertConfig, AdamW


# set argument parser
parser = argparse.ArgumentParser(description="For medical information analysis")

parser.add_argument('--data_path', required=False, default='data.csv')
parser.add_argument('--vocab_path', required=False, default='utils/vocab.txt')
parser.add_argument('--last_name', required=False, default=-1)

args = parser.parse_args()

# set vocabulary file and tokenizer
make_vocab.make_vocab(args.vocab_path)

# get data from csv file and set cuda or cpu
dp = data_provider.DataProvider(args.data_path, args.vocab_path)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# initialize model
config = BertConfig(vocab_size=3305, architectures=["BertForMaskedLM"])
model = BertForNextSentencePrediction(config=config)
model.to(device)
model.train()

train_loader = DataLoader(dp.train_dataset, batch_size=8, shuffle=True)
optim = AdamW(model.parameters(), lr=5e-5)
name = args.last_name

for epoch in range(100):
    losses = 0
    count = 0
    name += 1

    for batch in train_loader:
        optim.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        losses += loss
        count += 1

        loss.backward()
        optim.step()

    losses /= count
    print(f"{epoch} loss : {losses}")

    save_path = "./results/models/" + str(name) + ".pt"
    torch.save(model.state_dict(), save_path)

    with open("results/loss.txt", 'a+') as f:
        f.write(str(name) + " loss : " + str(losses.item()))

model.eval()

# initialize Trainer
# training_args = TrainingArguments(
#     output_dir='./results',          # output directory
#     num_train_epochs=100,            # total # of training epochs
#     per_device_train_batch_size=8,   # batch size per device during training
#     per_device_eval_batch_size=8,    # batch size for evaluation
#     warmup_steps=500,                # number of warmup steps for learning rate scheduler
#     weight_decay=0.01,               # strength of weight decay
#     logging_dir='./logs',            # directory for storing logs
# )
#
# trainer = Trainer(
#     model=model,                         # the instantiated 🤗 Transformers model to be trained
#     args=training_args,                  # training arguments, defined above
#     train_dataset=dp.train_dataset,      # training dataset
#     eval_dataset=dp.test_dataset,        # evaluation dataset
# )
#
# trainer.train()