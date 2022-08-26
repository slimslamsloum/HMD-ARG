"""

"""

import random
from typing import Counter

import numpy as np
import pytorch_lightning as pl
import torch
from biodatasets import list_datasets, load_dataset
from deepchain.models import MLP
from deepchain.models.utils import confusion_matrix_plot, model_evaluation_accuracy
from multitaskmodel_embedd import MultiTaskModelEmbedd
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch import Tensor, float32, float64
from torch.utils.data import DataLoader
from torchmetrics import F1, Accuracy, AveragePrecision, Precision

random.seed(10)

#TODO: this model has a pretty high variance: training and validation accuracy vary too much between
# different runs. It would be nice to fix this problem and have a high, stable accuracy.

# Load embedding and target dataset
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

# Compute the mechanisms associated to each protein
# There are 5 different resistance mechanisms
mechanism = mechanism[0][y == 1]
encoder_2 = preprocessing.LabelEncoder()
encoded_mechanism = encoder_2.fit_transform(mechanism)

# Compute embeddings (of size 1280)
cls_embeddings = np.load(
    "/home/selim/.cache/bio-datasets/antibiotic-resistance/sequence_esm1_t34_670M_UR100_cls_embeddings.npy",
    allow_pickle=True,
)[y == 1]

cls_embeddings = np.vstack(cls_embeddings).astype(np.float)

# Separate in training and test set. Compute dataloaders
x_train, x_test, y_train, y_test = train_test_split(
    cls_embeddings,
    np.column_stack((encoded_atb_classes, encoded_mechanism)),
    test_size=0.2,
    stratify=encoded_mechanism,
    random_state=10
)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=10)

batch_size = 32
trainloader = DataLoader(np.column_stack((x_train, y_train)), batch_size=batch_size)
valloader = DataLoader(np.column_stack((x_val, y_val)), batch_size=batch_size)
testloader = DataLoader(np.column_stack((x_test, y_test)), batch_size=batch_size)

#TODO: add model parameters such as scaling weights, learning rate, model architecture, etc.
# Right now these parameters are explicitly written in multitaskmodel_embedd.
# Create model
model = MultiTaskModelEmbedd()

# Train model
logger = TensorBoardLogger("tb_logs", name="my_model")
trainer = pl.Trainer(max_epochs=30, logger=logger)
trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=valloader)

trainer.test(model=model, dataloaders=testloader)
