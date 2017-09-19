import pandas as pd
import os
from datetime import datetime as dt
from numpy import unique, log2, log
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem.porter import *
stemmer = PorterStemmer()
stops = stopwords.words("english")


##define a function to get lexicon words
def getWords(filePath, encoding='utf-8'):
    with open(filePath, 'r', encoding=encoding) as ff:
        lines=ff.readlines()
        ff.close()
    words = list(set([i.strip() for i in lines]))
    return words

##Define the function Entropy

def word_entropy(s):
    res = 0
    val , counts = unique(s, return_counts=True)
    freqs = counts.astype('float')/len(s)
    freqs = [i*log2(i) for i in freqs if i > 0]
    entropy = -sum(freqs)
    return entropy

def binary_entropy(s):
    binary = ''.join(format(ord(x), 'b') for x in s)
    entropy = word_entropy(binary)
    return entropy

from textblob import TextBlob

def blobbed(s):
    blob = TextBlob(s)
    return blob

def polarity(s):
    blob = blobbed(s)
    polarity = blob.sentiment.polarity
    return polarity

def subjectivity(s):
    blob = blobbed(s)
    subjectivity = blob.sentiment.subjectivity
    return subjectivity

def sentiment(s):
    pol = polarity(s)
    sub = subjectivity(s)
    sentiment = (pol+sub)/2
    return sentiment

def word_count(s):
    counts = len(unique(s.split()))
    return counts

def word_len(s):
    length = len(s.split())
    return length



def context_relatedness(s,words):
    words = [stemmer.stem(i) for i in words if len(i) > 2 and i not in stops]
    counted = [s.count(i) for i in words if i in s]
    return sum(counted)
