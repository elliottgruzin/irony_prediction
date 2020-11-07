import collections
from nltk.tokenize import TweetTokenizer
import nltk
from nltk.corpus import stopwords

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

    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True).tokenize
    new_corpus = [tokenizer(sentence) for sentence in corpus]
    corpus = new_corpus
    return corpus

vocab = {}

dataset = "./train.txt".format(set)
corpus, y = parse_dataset(dataset)
corpus = featurize(corpus)
for i in range(len(corpus)):
    if y[i] == 0:
        line = corpus[i]
        for token in line:
            if token not in set(stopwords.words('english')):
                try:
                    vocab[token]+=1
                except:
                    vocab[token]=1

counts = collections.Counter(vocab)
most_common = counts.most_common(10)
print(most_common)
