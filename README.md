# HMD-ARG
This app is a base application that computes the probability that a protein is antibiotic-resistant (ARG).
The work is mainly based on the paper [HMD-ARG](https://microbiomejournal.biomedcentral.com/articles/10.1186/s40168-021-01002-3), which describes a more complex model that provides detailed annotations of ARGs.
We first use a bio-transformers with a 'esm1_t34_670M_UR100' model to compute the CLS embeddings. The dataset used is [antibiotic-resistance](https://github.com/DeepChainBio/bio-datasets/blob/main/datasets/antibiotic-resistance/description.md).

This app is a base application that uses a hierarchical multi-task method, HMD-ARG, which provides detailed annotations of antibiotic resistant genes (ARGs). The model first predicts if a given gene
will be an ARG or not (level 0), then it predicts the resistance mechanism and the antibiotic class it is resistant to (level 1), and finally if the predicted antbiotic family is beta-lactamase we predict the subclass of beta-lactamase the ARG is resistant to (level 2).

The work is mainly based on the paper [HMD-ARG](https://microbiomejournal.biomedcentral.com/articles/10.1186/s40168-021-01002-3). 

The following sketch describes the HDM-ARG:

## Dataset

The dataset has been curated by the Deepchain team on [antibiotic-resistance](https://github.com/DeepChainBio/bio-datasets/blob/main/datasets/antibiotic-resistance/description.md) and comprises of 17k samples. Each sample consists of a protein sequence, a label (0 or 1) indicating whether it's ARG or not,
the gene name, the antibiotic class it is resistant to and the resistance mechanism. 

Here is an example of a sample in our dataset: 

## Level 0

Here we seek to predict whether a protein is ARG or not.
We first apply a PCA to the embeddings with desired percentage of variance kept at 95%, which led to 
159 features (instead of the 1280 features of the embeddings). We then apply a T-SNE visualization to the
low-dimensional data, and notice the data classes are well separable:

![tsne.pgn](https://i.postimg.cc/SN7tY8sK/tsne.png)

The default model included is a MLP with input size 1280 (size of an embedding), two hidden dense layers, output size of 1 (binary classification into ARG or not-ARG), and ReLU activation. The model can be described by the following sketch:

![model.png](https://i.postimg.cc/tC0ZWYTZ/Screenshot-from-2022-07-20-13-29-22.png)

The model achieved 0.982 accuracy, 0.974 precision, 0.990 recall and 0.982 F1 Score on the Test set (with 30% of dataset in Test set and 70% in Training set).

The confusion matrix we obtained can be seen here:

![confusion-matrix.png](https://i.postimg.cc/85rWcpkz/confusion-matrix.png)

We notice our model did very well to separate the classes here. 

Logistic regression and SVM were also tried but the MLP provided the best overall results. 

## Level 1

If the protein is ARG, we then want to know what its resistance mechanism is and what antibiotic it is resistant to. To do so we use a multi-task learning deep neural network to predict at the same time
these two annotations. The model architecture is the following:

We achieved the following results:


## Level 2

Finally, if the protein is resistant to a beta-lactamase antibiotic, we predict what subclass of beta-lactamase it is resistant to. The subclass nomenclature is based on the website [beta-lactamase-database]https://ifr48.timone.univ-mrs.fr/beta-lactamase/public/. The model used is exactly the same as the one used in Level 0, and achieved the following results:





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

