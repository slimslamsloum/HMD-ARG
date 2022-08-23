import os

import numpy as np
import pandas as pd
import torch


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, word2id, max_len, data):
        self.word2id = word2id
        self.max_len = max_len

        self.sequences = data[:, :-2]
        self.atb_classes = data[:, -2:-1]
        self.mechanisms = data[:, -1:]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        seq = self.preprocess(self.sequences[index])
        return {
            "sequence": seq,
            "atb_class": self.atb_classes[index].astype(int),
            "mech": self.mechanisms[index].astype(int),
        }

    def preprocess(self, text):
        seq = []

        # Encode into IDs
        for word in text[: self.max_len]:
            seq.append(self.word2id.get(word, self.word2id["<unk>"]))

        # Pad to maximal length
        if len(seq) < self.max_len:
            seq += [self.word2id["<pad>"] for _ in range(self.max_len - len(seq))]

        # Convert list into tensor
        seq = torch.from_numpy(np.array(seq))

        # One-hot encode
        one_hot_seq = torch.nn.functional.one_hot(
            seq,
            num_classes=len(self.word2id),
        )

        # Permute channel (one-hot) dim first
        one_hot_seq = one_hot_seq.permute(1, 0)

        return one_hot_seq
