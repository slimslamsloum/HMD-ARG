import random
from typing import Counter

import numpy as np
import torch
from biodatasets import list_datasets, load_dataset
from sklearn import preprocessing

dataset = load_dataset("antibiotic-resistance")

X, y = dataset.to_npy_arrays(input_names=["sequence"], target_names=["label"])
X, atb_classes = dataset.to_npy_arrays(
    input_names=["sequence"], target_names=["antibiotic_class"]
)
X, mechanism = dataset.to_npy_arrays(
    input_names=["sequence"], target_names=["mechanism"]
)
y = y[0]

# Compute atb classes and encode them.
atb_classes = atb_classes[0][y == 1]
encoder_1 = preprocessing.LabelEncoder()
encoded_atb_classes = encoder_1.fit_transform(atb_classes)
# print(len((encoded_atb_classes)))

# Compute the mechanisms associated to each protein
# There are 5 different resistance mechanisms
mechanism = mechanism[0][y == 1]
encoder_2 = preprocessing.LabelEncoder()
encoded_mechanism = encoder_2.fit_transform(mechanism)
# print(len((encoded_mechanism)))

# Compute scaling weights for classes (since our data is imbalanced)
counter_atb = Counter(encoded_atb_classes)
counter_mech = Counter(encoded_mechanism)

weights_atb = torch.zeros([22])
weights_mech = torch.zeros([5])


for k in counter_atb.keys():
    weights_atb[k] = len(encoded_atb_classes)/(22*counter_atb[k])

for k in counter_mech.keys():
    weights_mech[k] = len(encoded_mechanism)/(5*counter_mech[k])