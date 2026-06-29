import numpy as np
import pandas as pd 

train = pd.read_csv("train.tsv", sep="\t", header = None)
valid = pd.read_csv("valid.tsv", sep="\t", header = None)
test = pd.read_csv("test.tsv", sep="\t", header = None)
graph = pd.concat([train, valid, test])

train_sort = sorted(set(train[0]) | set(train[2]))
test_sort = sorted(set(test[0]) | set(test[2]))
valid_sort = sorted(set(valid[0])| set(valid[2]))
entities = sorted(set(graph[0]) | set(graph[2]))

print(len(train_sort))
print(len(test_sort))
print(len(valid_sort))
print(len(entities))
bert = np.load("pubmedbert_embeddings_768.npy")
poin = np.load("poincare_embeddings.npy")

print(bert.shape)
print(poin.shape)