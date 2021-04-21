from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import RobertaTokenizerFast
from transformers import RobertaConfig
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import RobertaForMaskedLM
from transformers import Trainer, TrainingArguments


def train_tok(txt_dir, tokenizer_dir):
    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train(files=txt_dir, vocab_size=52_000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    tokenizer.save_model(tokenizer_dir)


def get_tok(tokenizer_dir):
    return RobertaTokenizerFast.from_pretrained(tokenizer_dir, max_len=512)


def train_mod(txt_dir, tokenizer, model_dir):
    config = RobertaConfig(
        vocab_size=3305,
        max_position_embeddings=1024,
        num_attention_heads=12,
        num_hidden_layers=6,
        output_attentions=True,
        type_vocab_size=1,
    )

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path=txt_dir,
        block_size=1024
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    model = RobertaForMaskedLM(config=config)

    training_args = TrainingArguments(
        output_dir=model_dir,
        overwrite_output_dir=True,
        num_train_epochs=1000,
        per_gpu_train_batch_size=16,
        save_steps=1000,
        save_total_limit=37,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    trainer.train()

    trainer.save_model(model_dir)


def model_sequence(txt_dir, tokenizer_dir, model_dir, is_tok_train=False, is_mod_train=False):
    '''
    :param txt_dir: train, text txt 파일이 저장되어 있는 장소
    :param tokenizer_dir: tokenizer가 저장된 장소
    :param model_dir: 모델이 저장된 장소
    :param is_tok_train: tokenizer 학습 할거?
    :param is_mod_train: model 학습 할거?
    '''
    if is_tok_train:
        tokenizer = train_tok(txt_dir, tokenizer_dir)

    if is_mod_train:
        tokenizer = get_tok(tokenizer_dir)
        train_mod(txt_dir, tokenizer, model_dir)
