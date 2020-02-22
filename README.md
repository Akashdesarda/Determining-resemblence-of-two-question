# Determining resemblance of two question

---

Deep learning tool to determine similarity of given two question. The tool will is able to understand context of question to determine the similarity.

>eg:
Question 1: What is your name?
Question 2: Who are you?
--> They are Similar 

### Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Performance](#performance)
4. [Examples](#examples)
5. [How to use it](#how-to-use-it)
6. [Download the Weights and embeddings](#download-the-weights-and-embeddings)
7. [How to fine-tune one of the trained models on your own dataset](#how-to-fine-tune-one-of-the-trained-models-on-your-own-dataset)
8. [ToDo](#todo)


### Overview

This is two part Deep learning based NLP solution to determine resemblance or similarity of two given question.

Part one is to extract word embeddings using [**_Universal Word Embedding_**](https://tfhub.dev/google/collections/universal-sentence-encoder/1) and second part is to predict wether the given two question are similair or not.

The main goal of this project is to try alternative against some traditional approach like *TFIDF*, *Minimum distance*, *Co-Occurrence Matrix*, or even *Word2Vec*, *GolVe*, etc and achieve state-of-the-art result.

[Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs) dataset have been used to train data.The repository will also provide pretrained weights and word embeddings(in .npy format).

### Architecture

The architecture can be divided in two part:

1. Embeddings generation:

- Two option were explored, *Word2Vec* & *Universal sentence encoder* & the latter was chosen. *Word2Vec* generate embeddings per word which make it more difficult for the network to the overall context of sentence (here question). *Universal sentence encoder* generates embedding of complete sentence, which helps network to understand overall context of question.

- *Universal sentence encoder* generates a output of vector size of 512 whatever may be the input single word, a sentence or even a small paragraph (there in no fixed size on paragraph).

- A direct consumable form of *Universal sentence encoder* is available at [Tensorflow hub](https://tfhub.dev/google/universal-sentence-encoder-large/5). More details are mentioned at its page.

2. Prediction using SimilarityNet:

- The Network is three layer & finally softmax layer to predict.

### Performance

### Examples

### How to use it

This repository provides Jupyter notebook tutorial that explain training, inference and evaluation, and there are a bunch of explanations in the subsequent sections that complement the notebooks.

#### Training Details

Training could be performed in two steps:

1. Generation of embeddings using any desired text embedding method.
2. Using the text embedding to feed to Network and train the model.

#### Dataset

Quora Question Pairs dataset was used to train & it can be downloaded from [here](https://www.kaggle.com/c/quora-question-pairs).

**Note:** Any other dataset will also do.

### Download the Weights and embeddings

- Text embeddings generated from Universal sentence encoder for Quora Question Pairs can be downloaded from here
- Weights can be downloaded from here

### How to fine-tune one of the trained models on your own dataset

Fine-tuning can be done by changing two configration.

1. Change text embedding generator

2. Change Hidden layer configration

### TODO

