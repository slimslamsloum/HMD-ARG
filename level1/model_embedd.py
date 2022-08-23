"""

"""

import random
from typing import Counter

import numpy as np
import torch
from biodatasets import list_datasets, load_dataset
from deepchain.models import MLP
from deepchain.models.utils import (confusion_matrix_plot,
                                    model_evaluation_accuracy)
from multitaskmodel_embedd import MultiTaskModelEmbedd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch import Tensor, float32, float64
from torch.utils.data import DataLoader
from torchmetrics import F1, Accuracy, AveragePrecision, Precision, Recall

random.seed(10)

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
# print(len((encoded_atb_classes)))

# Compute the mechanisms associated to each protein
# There are 5 different resistance mechanisms
mechanism = mechanism[0][y == 1]
encoder_2 = preprocessing.LabelEncoder()
encoded_mechanism = encoder_2.fit_transform(mechanism)
# print(len((encoded_mechanism)))


# Compute embeddings (of size 1280)
cls_embeddings = np.load(
    "/home/selim/.cache/bio-datasets/antibiotic-resistance/sequence_esm1_t34_670M_UR100_cls_embeddings.npy",
    allow_pickle=True,
)[y == 1]

cls_embeddings = np.vstack(cls_embeddings).astype(np.float)

counter_atb = Counter(encoded_atb_classes)
counter_mech = Counter(encoded_mechanism)

weights_atb = torch.zeros([22])
weights_mech = torch.zeros([5])


for k in counter_atb.keys():
    weights_atb[k] = len(encoded_atb_classes)/(22*counter_atb[k])

for k in counter_mech.keys():
    weights_mech[k] = len(encoded_mechanism)/(22*counter_mech[k])

print(weights_atb)
print(weights_mech)

# Separate in training and test set. Compute dataloaders
x_train, x_test, y_train, y_test = train_test_split(
    cls_embeddings,
    np.column_stack((encoded_atb_classes, encoded_mechanism)),
    test_size=0.3,
)
batch_size = 32
trainloader = DataLoader(np.column_stack((x_train, y_train)), batch_size=batch_size)
testloader = DataLoader(np.column_stack((x_test, y_test)), batch_size=batch_size)

# Create model
model = MultiTaskModelEmbedd()

# Train model
model.training_step(
    trainloader=trainloader, nb_epochs=10, atb_coeff=0.1, mech_coeff=1.2
)

# Save model
model.save_model(path="/home/selim/Documents/myApps/HMD-ARG/level1/model_embedd")

# Predict on test set
atb_pred, mech_pred = model.forward(x_test)

# Append the atb and mechanism predictions into a list
pred = []
for i in range(len(x_test)):
    pred.append(
        (
            np.argmax(atb_pred[i].detach().numpy()),
            np.argmax(mech_pred[i].detach().numpy()),
        )
    )

# Accuracy evaluation
accuracy = Accuracy(mdmc_average="global")
print("Accuracy: {0}".format(accuracy(torch.tensor(pred), torch.from_numpy(y_test))))

# Precision evaluation
precision = Precision(mdmc_average="global")
print(
    "Precision: {0}".format(
        precision(torch.tensor(pred), torch.from_numpy(y_test).int())
    )
)

# Recall evalution
recall = Recall(mdmc_average="global")
print("Recall: {0}".format(recall(torch.tensor(pred), torch.from_numpy(y_test).int())))

# F1 Score evaluation
f1score = F1(mdmc_average="global")
print(
    "F1 Score: {0}".format(f1score(torch.tensor(pred), torch.from_numpy(y_test).int()))
)
