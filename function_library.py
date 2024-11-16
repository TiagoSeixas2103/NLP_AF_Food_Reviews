import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import urllib.request
import zipfile
import re
import os

class ZipFileWithProgress(zipfile.ZipFile):
    def extractall(self, path=None, members=None, pwd=None):
        if members is None:
            members = self.namelist()
        total = len(members)
        
        with tqdm(total=total, unit='file') as pbar:
            for member in members:
                self.extract(member, path, pwd)
                pbar.update(1)

def get_vocabulary(text : str,
                   expr: str=r"\b\w+\b",
                   case_sensitive : bool=False,
                   ) -> dict:
    if case_sensitive == False:
        text = text.lower()  
    vocabulary = set(re.findall(expr, text))
    vocab = dict()
    inverse_vocab = list()
    vocab["<PAD>"] = 0
    inverse_vocab.append("<PAD>")
    vocab["<UNK>"] = 1
    inverse_vocab.append("<UNK>")
    for i, token in enumerate(vocabulary):
        if token not in vocab:
            vocab[token] = i+2 # We start from 2 because 0 and 1 are reserved for <UNK> and <PAD>
            inverse_vocab.append(token)
    return vocab, inverse_vocab

def tokenize_words(text : str,
             vocab : dict,
             expr : str= r"\b\w+\b",
             sentence_length : int = 10,
             case_sensitive : bool = False) -> list:
    if case_sensitive == False:
        text = text.lower()
    words = re.findall(expr, text)
    tokens = []
    for i, w in enumerate(words):
        if i == sentence_length:
            break
        if w in vocab:
            tokens.append(vocab[w])
        else:
            tokens.append(vocab["<UNK>"])


    if len(tokens) < sentence_length:
        n_pad = sentence_length - len(tokens)
        pad = [vocab["<PAD>"]] * n_pad
        tokens = pad + tokens
    return tokens

def detokenize_words(tokens : list,
                    invert_vocab : list) -> str:
    text = " ".join([invert_vocab[token] for token in tokens])
    return text

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer(text)
        x = torch.tensor(tokens)
        y = torch.tensor(label).reshape(1).clone().detach()
        return x, y
    
class MyTokenizer:
    def __init__(self, sentence_length, case_sensitive=False):
        self.sentence_length = sentence_length
        self.case_sensitive = case_sensitive

    def fit(self, phrases : list, expr : str=r"\b\w+\b"):
        self.vocab, self.inverse_vocab = get_vocabulary(" ".join(phrases),
                                                        expr=expr,
                                                        case_sensitive=self.case_sensitive)
        self.vocab_size = len(self.vocab)
        
    def __call__(self, x):
        return tokenize_words(x,
                              self.vocab,
                              sentence_length=self.sentence_length,
                              case_sensitive=self.case_sensitive)

def load_glove_vectors(glove_file):
    glove_vectors = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = torch.tensor([float(val) for val in values[1:]], dtype=torch.float32)
            glove_vectors[word] = vector
    return glove_vectors

def get_vocabulary_from_glove(glove_vectors):
    vocab = dict()
    inverse_vocab = list()
    vocab["<PAD>"] = 0
    inverse_vocab.append("<PAD>")
    vocab["<UNK>"] = 1
    inverse_vocab.append("<UNK>")
    for word, vector in glove_vectors.items():
        vocab[word] = len(inverse_vocab)
        inverse_vocab.append(word)
    return vocab, inverse_vocab

class MyOtherClassifier( nn.Module ):
    def __init__(self, vocab_size, embedding_layer, embedding_dim, hidden_dim, output_dim, n_special_tokens=2, n_layers=10):
        super(MyOtherClassifier, self).__init__()
        self.n_special_tokens = n_special_tokens
        self.embedding = embedding_layer
        self.hidden = hidden_dim
        self.fc = nn.Linear(embedding_dim, hidden_dim)
        self.fc_multi = self.multi_layered(n_layers)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def _pool(self, x):
        x = torch.mean(x, dim=1)
        return x

    def multi_layered(self, n):
        layers = nn.Sequential()
        for i in range(n):
            layers.add_module(f'layer{i}', nn.Linear(self.hidden, self.hidden))
        return layers

    def forward(self, x):
        x = self.embedding(x)
        x = self._pool(x)
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc_multi(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x