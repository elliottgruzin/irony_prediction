#!/usr/bin/env python3

'''
example.py

Benchmark system for the SemEval-2018 Task 3 on Irony detection in English tweets.
The system makes use of token unigrams as features and outputs cross-validated F1-score.

Date: 1/09/2017
Copyright (c) Gilles Jacobs & Cynthia Van Hee, LT3. All rights reserved.
'''

from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.datasets import dump_svmlight_file
from sklearn import metrics
import numpy as np
import logging
import codecs

## Additional packages used by Elliott Gruzin

from bert_serving.client import BertClient
from torch.utils.data import TensorDataset
import torch
import pickle
from transformers import AutoTokenizer ## from hugging face
from nltk.sentiment.vader import SentimentIntensityAnalyzer

logging.basicConfig(level=logging.INFO)

def parse_dataset(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    y = []
    corpus = []
    with open(fp, 'rt') as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                label = int(line.split("\t")[1])
                tweet = line.split("\t")[2]
                y.append(label)
                corpus.append(tweet)

    return corpus, y


def featurize(corpus):
    '''
    Tokenizes and creates sentence vectors.
    :param corpus: A list of strings each string representing document.
    :return: X: List of BERT-embedded sentences, as well as retokenized corpus.
    '''
    sentiments = []
    analyser = SentimentIntensityAnalyzer()
    for sentence in corpus:
        sentiment_values = np.array(list(analyser.polarity_scores(sentence).values()))
        sentiments.append(sentiment_values)
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize
    new_corpus = [' '.join(tokenizer(sentence)) for sentence in corpus]
    corpus = new_corpus
    bc = BertClient()
    X = bc.encode(corpus)
    return X, sentiments



if __name__ == "__main__":

    # Create and store BERT Embeddings

    # STEP 1: Make dictionaries to identify samples, and give label

    partition_dict = {'train':[],'test':[]}
    label_dict = {}

    # STEP 2: Compute and store

    for set in ['train','test']:

        dataset = "./{}_no_hashtag.txt".format(set)
        corpus, y = parse_dataset(dataset)
        print('Extracting BERT embeddings for each tweet...')
        X, sentiments = featurize(corpus)
        print('Embeddings extracted. Storing embeddings...')
        for i in range(len(y)):
            id = set+str(i)
            partition_dict[set].append(id)
            label_dict[id] = y[i]
            sentence_and_sentiment = np.append(X[i],sentiments[i])
            embedding = torch.from_numpy(sentence_and_sentiment)
            torch.save(embedding, 'data/dehashtagged/{}.pt'.format(id))
        print('Embeddings stored for the {} dataset.\n'.format(set))

    print('Writing dictionaries...')

    p_dic = open('data/dehashtagged/partition_dict.pkl','wb')
    l_dic = open('data/dehashtagged/label_dict.pkl','wb')
    pickle.dump(partition_dict, p_dic)
    pickle.dump(label_dict, l_dic)
    p_dic.close()
    l_dic.close()
