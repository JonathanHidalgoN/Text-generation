import torch
from torch.utils.data import Dataset, random_split
from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPT2LMHeadModel
from utils.read_text import read_text


class Tweets(Dataset):
    """
    A dataset of tweets.
    Args:
        txt_list: A list of tweets.
        tokenizer: The tokenizer to use.
        max_length: The maximum length of a tweet.
    Inherits from: torch.utils.data.Dataset
    """

    def __init__(self, txt_list, tokenizer, max_length):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        for txt in txt_list:
            encodings_dict = tokenizer(
                "<|startoftext|>" + txt + "<|endoftext|>",
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            self.input_ids.append(torch.tensor(encodings_dict["input_ids"]))
            self.attn_masks.append(torch.tensor(encodings_dict["attention_mask"]))

    def __len__(self):
        """
        Custom method to get the length of the dataset.
        """
        return len(self.input_ids)

    def __getitem__(self, idx):
        """
        Custom method to get a sample from the dataset.
        """
        return self.input_ids[idx], self.attn_masks[idx]


if __name__ == "__main__":

    # This is runing in a kaggle notebook, you may need to change the path to the data or the parameters.
    # Define parameters
    token_parameters = {
        "model": "gpt2-medium",
        "bos_token": "<|startoftext|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "<|pad|>",
    }
    data_parameters = {
        "path": "data_path",
        "delimiter": "|-|",
        "to_read": 10000,
        "training_size": 0.9,
    }
    training_parameters = {
        "output_dir": "./results",
        "num_train_epochs": 1,
        "logging_steps": 100,
        "save_steps": 5000,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "warmup_steps": 10,
        "weight_decay": 0.05,
        "logging_dir": "./logs",
        "report_to": "none",
    }

    generation_parameters = {
        "do_sample": True,
        "top_k": 50,
        "max_length": 300,
        "top_p": 0.95,
        "temperature": 1.9,
        "num_return_sequences": 20,
    }

    tokenizer = GPT2Tokenizer.from_pretrained(
        token_parameters["model"],
        bos_token=token_parameters["bos_token"],
        eos_token=token_parameters["eos_token"],
        pad_token=token_parameters["pad_token"],
    )

    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(token_parameters["model"]).cuda()
    model.resize_token_embeddings(len(tokenizer))
    # Load data
    tweets = read_text(
        filename=data_parameters["path"],
        delimiter=data_parameters["delimiter"],
        lines_to_return=data_parameters["to_read"],
    )
    max_length = max([len(tokenizer.encode(tweet)) for tweet in tweets])

    # Deine the data as Tweets class because torch needs it as a Dataset
    dataset = Tweets(tweets, tokenizer, max_length=max_length)
    # Split the data into training and validation
    train_size = int(data_parameters["training_size"] * len(dataset))
    train_dataset, val_dataset = random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    # Define the training arguments as a TrainingArguments class because torch needs it as a TrainingArguments
    training_args = TrainingArguments(
        output_dir=training_parameters["output_dir"],
        num_train_epochs=training_parameters["num_train_epochs"],
        logging_steps=training_parameters["logging_steps"],
        save_steps=training_parameters["save_steps"],
        per_device_train_batch_size=training_parameters["per_device_train_batch_size"],
        per_device_eval_batch_size=training_parameters["per_device_eval_batch_size"],
        warmup_steps=training_parameters["warmup_steps"],
        weight_decay=training_parameters["weight_decay"],
        logging_dir=training_parameters["logging_dir"],
        report_to=training_parameters["report_to"],
    )

    # Define the trainer and train the model
    Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda data: {
            "input_ids": torch.stack([f[0] for f in data]),
            "attention_mask": torch.stack([f[1] for f in data]),
            "labels": torch.stack([f[0] for f in data]),
        },
    ).train()

    generated = tokenizer(
        token_parameters["bos_token"], return_tensors="pt"
    ).input_ids.cuda()

    # Generate tweets
    sample_outputs = model.generate(
        generated,
        generation_parameters["do_sample"],
        top_k=generation_parameters["top_k"],
        max_length=300,
        top_p=0.95,
        temperature=generation_parameters["temperature"],
        num_return_sequences=generation_parameters["num_return_sequences"],
    )

    # Save the generated tweets
    samples = [
        tokenizer.decode(output, skip_special_tokens=True) for output in sample_outputs
    ]
