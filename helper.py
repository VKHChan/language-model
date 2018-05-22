# -*- coding: utf-8 -*-
"""
helper functions

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import warnings
import re

def pickle_object(obj, name):
    with open(name+".pkl", 'wb') as f:
        pickle.dump(obj, f, protocol=4)


def unpickle_object(pkl):
    return pickle.load(open(pkl, 'rb'))

'''
Functions to clean the data
clean1: remove all punctuations and repeated characters
'''
def clean1(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    #string = re.sub(r"\'s", " \'s", string) 
    #string = re.sub(r"\'ve", " \'ve", string) 
    #string = re.sub(r"n\'t", " n\'t", string) 
    #string = re.sub(r"\'re", " \'re", string) 
    #string = re.sub(r"\'d", " \'d", string) 
    #string = re.sub(r"\'ll", " \'ll", string)
    #string = re.sub(r"\'m", " \'m", string)
    #string = re.sub(r",", " , ", string) 
    #string = re.sub(r"\.{2,}", " .", string)
    #string = re.sub(r"\.", " \. ", string)
    #string = re.sub(r"!", " ! ", string) 
    #string = re.sub(r"\(", " \( ", string) 
    #string = re.sub(r"\)", " \) ", string) 
    #string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip()

# given a transript (a paragraph of words) and number of words in the sequence
# return the organize sequence of words
def create_input_output(transcript, n_seq = 51):
    transcript = transcript.split()
    sequences = list()
    for i in range(n_seq, len(transcript)):
        seq = transcript[i-n_seq:i]
        line = " ".join(seq)
        sequences.append(line)
    print('Total Sequences: %d' % len(sequences))
    return sequences
