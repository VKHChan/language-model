'''
Prepare data for training
- Read data from file
- Cleaning data
- Generating tokenizer (pickle tokenizer)
- Split dataset (pickle split dataset)

- Generate sequences for training and vaidation data
- Encode training and validation data (pickle)
'''

from helper import *

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from collections import Counter
import random
import re
import gc

from keras.preprocessing.text import Tokenizer

if __name__ == "__main__":
    
# TO DO: argparse some of the parameters such as n_words, n_seq

    # read data from xml file
    tree = ET.parse('ted_en-20160408.xml')
    root = tree.getroot()
    # get all content
    all_transcript = [root[i][1].text for i in range(len(root))]

    clean_transcript = [clean1(transcript) for transcript in all_transcript]

    # split data to training, validation and testing
    train = clean_transcript[:1585]
    valid = clean_transcript[1585:1835]
    test = clean_transcript[1835:]

    pickle_object(train, "train_clean")
    pickle_object(valid, "valid_clean")
    pickle_object(test, "test_clean")

    n_words= 20000

    # assign "UNK" as oov_token
    tokenizer = Tokenizer(oov_token="UNK", num_words=n_words+1)
    # fit tokenizer on training text
    tokenizer.fit_on_texts(train)
    # modify word_index so that all vocabs not found is assigned "UNK" token
    tokenizer.word_index = {e:i for e,i in tokenizer.word_index.items() if i <= n_words} # <= because tokenizer is 1 indexed
    tokenizer.word_index[tokenizer.oov_token] = n_words + 1
    pickle_object(tokenizer, "tokenizer")
    
    # vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    print("vocab_size = ", vocab_size)

    # join all transcript together to one corpus
    train_corpus = " ".join(train)
    valid_corpus = " ".join(valid)

    # number of words in each sequence
    # i.e. model will be learning from the first (n_seq-1) of words in the sequence to predict the last word
    n_seq = 21
    train_seq = create_input_output(transcript=train_corpus, n_seq=n_seq)
    valid_seq = create_input_output(transcript=valid_corpus, n_seq=n_seq)

    # encode the train_seq
    train_seq_encode = tokenizer.texts_to_sequences(train_seq)
    # encode the valid_seq
    valid_seq_encode = tokenizer.texts_to_sequences(valid_seq)

    pickle_object(train_seq_encode, "train_seq_encode")
    pickle_object(valid_seq_encode, "valid_seq_encode")
