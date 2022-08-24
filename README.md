# HMD-ARG

The rapid spread of antibiotic resistant bacteria is occuring worldwide, and is endangering global health by mitigating the efficacy of antibiotics, which have transformed medicine and saved millions of lives. This resistance is due to the overuse and misuse of antibiotics, which has led to Antibiotic Resistant Genes (ARG). Accurately identifying and understanding ARGs is an indispensable step to solve the antibiotic resistance crisis, which is the goal of this project. 

This project is a base application that uses a hierarchical multi-task method, HMD-ARG, which provides detailed annotations of ARGS. The model is divided into three seperate levels. Namely, the model first predicts if a given protein
is produced by an ARG or not (Level 0). If it is, then the model predicts the resistance mechanism and the antibiotic class the gene is resistant to (Level 1). And finally, if the predicted antibiotic family is beta-lactamase, we predict the subclass of beta-lactamase the ARG is resistant to (level 2).

The work is mainly based on the paper [HMD-ARG](https://microbiomejournal.biomedcentral.com/articles/10.1186/s40168-021-01002-3). 

The following sketch describes HMD-ARG:

![hmd-arg.png](https://i.postimg.cc/MTy2J9TH/hmd-arg.png)

## Dataset

The dataset has been curated by the Deepchain team on [antibiotic-resistance](https://github.com/DeepChainBio/bio-datasets/blob/main/datasets/antibiotic-resistance/description.md) and comprises of 17k samples. Each sample consists of a protein sequence, a label (0 or 1) indicating whether it has been produced by an ARG or not, the gene name, the antibiotic class it is resistant to and the resistance mechanism. 

Here is an example of a sample in our dataset: 

    Protein_ID: ACN5894.1

    Sequence: MAEPVLSVKD...

    antibiotic_class: macrolide-lincosamide-streptogramin

    gene_name: macB

    mechanism: antibiotic target prediction

    label: 0.0 (which means it isn't from an ARG)

## Level 0

Here we seek to predict whether a protein is produced by an ARG or not. 
Our raw protein sequences are first embedded using a Transformer, with each embedding having 1280 features.
We then apply a dimensionality reduction method, namely PCA, to these embeddings with desired percentage of variance kept at 95%, which led to 
samples with 159 features. We then apply the T-SNE method to our data to visualize it in 2 dimensions. We clearly notice that thanks to the Transformer and the use of PCA, we have only kept relevant information and separated the data into 2 classes quite nicely.

![tsne.pgn](https://i.postimg.cc/SN7tY8sK/tsne.png)

The default model included is a MLP with input size 1280 (size of an embedding), two hidden dense layers, output size of 1 (binary classification into ARG or not-ARG), and ReLU activation. The model can be described by the following sketch:

![model.png](https://i.postimg.cc/tC0ZWYTZ/Screenshot-from-2022-07-20-13-29-22.png)

The model achieved 0.982 accuracy, 0.974 precision, 0.990 recall and 0.982 F1 Score on the Test set (with 30% of dataset in Test set and 70% in Training set).

The confusion matrix we obtained can be seen here:

![confusion-matrix.png](https://i.postimg.cc/85rWcpkz/confusion-matrix.png)

We notice our model did very well to separate the classes here. 

Logistic regression and SVM were also tried but the MLP provided the best overall results. 

## Level 1

If the protein is indeed coming from an ARG, we then want to know what its resistance mechanism is and what antibiotic it is resistant to. To do so we use a multi-task learning deep neural network to predict at the same time
these two labels (the resistance mecanism, and the antibiotic class). The model architecture is the following:

![level1](https://i.postimg.cc/VvcG91zL/level1.png)

We achieved the following results:


## Level 2

Finally, if the protein is resistant to a beta-lactamase antibiotic, we predict what subclass of beta-lactamase it is resistant to. The subclass nomenclature is based on the website [beta-lactamase-database](https://ifr48.timone.univ-mrs.fr/beta-lactamase/public/). The model used is exactly the same as the one used in Level 0, and achieved the following results:

Accuracy: 0.92, Precision: 0.92, Recall: 0.92, F1: 0.91.

## libraries
- pytorch>=1.5.0
- numpy
- pandas
- torch
- sklearn
- biotransformers
- biodatasets
- deepchain

## tasks
- transformers
- pca
- binary classification
- multi-task learning

## embeddings
- ESM1_t34_670M_UR100

## Author

Selim Jerad - Research Intern @ InstaDeep, Bachelor Student @ EPFL - s.jerad@instadeep.com

## Datasets / Resources

[HMD-ARG](https://microbiomejournal.biomedcentral.com/articles/10.1186/s40168-021-01002-3)  

[antibiotic-resistance](https://github.com/DeepChainBio/bio-datasets/blob/main/datasets/antibiotic-resistance/description.md)

[deepchain apps](https://github.com/DeepChainBio/deepchain-apps)

