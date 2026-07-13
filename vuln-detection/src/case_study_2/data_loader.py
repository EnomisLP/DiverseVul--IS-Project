import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

EMPTY_CODE_SENTINEL = "EMPTY_ABSTRACTED_CODE_SAMPLE"

class DiverseVulDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512, code_column="normalized_code"):
        self.df = dataframe.copy().reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.code_column = code_column
        
        self.df[self.code_column] = self.df[self.code_column].fillna("").astype(str)
        empty_mask = self.df[self.code_column].str.strip().eq("")
        if empty_mask.any():
            self.df.loc[empty_mask, self.code_column] = EMPTY_CODE_SENTINEL

        self.labels = self.df["label"].values.astype(np.float32)
        self.codes = self.df[self.code_column].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        code_text = self.codes[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            code_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.float32)
        }

def get_class_weights(dataframe):
    labels = dataframe["label"].values.astype(int)
    neg_counts = np.sum(labels == 0)
    pos_counts = np.sum(labels == 1)
    
    if pos_counts == 0:
        return torch.tensor([1.0])
        
    pos_weight = neg_counts / pos_counts
    return torch.tensor([pos_weight], dtype=torch.float32)

def create_dataloader(dataframe, tokenizer, batch_size=16, max_length=512, shuffle=True, code_column="normalized_code"):
    dataset = DiverseVulDataset(
        dataframe=dataframe, 
        tokenizer=tokenizer, 
        max_length=max_length, 
        code_column=code_column
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=False
    )
    return dataloader