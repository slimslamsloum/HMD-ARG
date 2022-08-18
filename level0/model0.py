"""
Module that provides a classifier to train a model on embeddings and predict
whether a protein is ARG or not. The dataset used is the antibiotic-resistance
from biodatasets, and the embedding of the 17k proteins come from the 
esm1_t34_670M_UR100 model.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from biodatasets import list_datasets, load_dataset
from deepchain.models import MLP
from deepchain.models.utils import (confusion_matrix_plot,
                                    model_evaluation_accuracy)
from joblib import dump, load
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import Tensor, float32, float64, norm, randperm
from torchmetrics import F1, Accuracy, AveragePrecision, Precision, Recall

# Load embedding and target dataset
dataset = load_dataset("antibiotic-resistance")

X, y = dataset.to_npy_arrays(input_names=["sequence"], target_names=["label"])

cls_embeddings = np.load(
    "/home/selim/.cache/bio-datasets/antibiotic-resistance/sequence_esm1_t34_670M_UR100_cls_embeddings.npy",
    allow_pickle=True)

cls_embeddings = torch.tensor(np.vstack(cls_embeddings).astype(np.float))

# Normalize the embeddings
normalized = StandardScaler().fit_transform(cls_embeddings)

# Apply PCA to normalized embeddings with 95% kept variance which yields 159 components.
pca = PCA(n_components=0.95)
pca = pca.fit(normalized)
lowdim_embeddings = pca.transform(normalized)

# Compute and store a t-sne visualization of the low-dim embeddings with 2 components.
x_tsne = TSNE().fit_transform(lowdim_embeddings)
df = pd.DataFrame()
df["tsne-2d-one"] = x_tsne[:,0]
df["tsne-2d-two"] = x_tsne[:,1]
df["target"] = y[0]

plt.figure(figsize=(16,10))

sns.scatterplot(
    data = df,
    x="tsne-2d-one", y="tsne-2d-two",
    hue ="target",
    palette=sns.color_palette("hls",2),
    legend="full",
    alpha =0.3
)

# Store tsne plot and pca model
#plt.savefig('src/level0/tsne.png')
#dump(pca, 'src/level0/pca.joblib')

#Separate data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(np.array(lowdim_embeddings), y[0], test_size=0.3)


#---------------------------------Logistic Regression-------------------------------------------------

#Create logreg model, compute predictions on test set and store model
#logreg = LogisticRegression(random_state=0, solver='liblinear').fit(x_train, y_train)
#y_pred = logreg.predict_proba(x_test)
#y_pred = np.array([x[1] for x in y_pred])



#-----------------------------------------------------------------------------------------------------

#---------------------------------SVM-------------------------------------------------

#Create svm model, compute predictions on test set and store model
#svm = SVC(probability=True).fit(x_train, y_train)
#y_pred = svm.predict_proba(x_test)
#y_pred = np.array([x[1] for x in y_pred])
#dump(svm, 'src/level0/svm0.joblib')


#-----------------------------------------------------------------------------------------------------

# Build a multi-layer-perceptron on top of embedding

# The fit method can handle all the arguments available in the
# 'trainer' class of pytorch lightening :
#               https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html
# Example arguments:
# * specifies all GPUs regardless of its availability :
#               Trainer(gpus=-1, auto_select_gpus=False, max_epochs=20)

#---------------------------------MLP-----------------------------------------------------------------

n_class = len(np.unique(y_train))
#print(n_class)
input_shape = x_train.shape[1]
#print(input_shape)

mlp = MLP(input_shape=input_shape, n_class=n_class)
mlp.fit(x_train, y_train, epochs=16)
mlp.save("model.pt")

# Model evaluation
y_pred = mlp(x_test).squeeze().detach().numpy()
print(y_pred)
model_evaluation_accuracy(y_test, y_pred)

#------------------------------------------------------------------------------------------------------

# Plot confusion matrix
confusion_matrix_plot(y_test, (y_pred > 0.5).astype(int), ["0", "1"])

# Accuracy evaluation
accuracy = Accuracy()
print('Accuracy: {0}'.format(accuracy(torch.from_numpy(y_pred), torch.from_numpy(y_test).int())))

# Precision evaluation
precision = Precision()
print('Precision: {0}'.format(precision(torch.from_numpy(y_pred), torch.from_numpy(y_test).int())))

# Recall evalution
recall = Recall()
print('Recall: {0}'.format(recall(torch.from_numpy(y_pred), torch.from_numpy(y_test).int())))

# F1 Score evaluation
f1score = F1()
print('F1 Score: {0}'.format(f1score(torch.from_numpy(y_pred), torch.from_numpy(y_test).int())))








