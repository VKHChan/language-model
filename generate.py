'''
TO generate text from train models
'''

from helper import *
from test import *

import numpy as np
import pandas as pd
from keras.models import load_model
from keras import backend as K
from keras.preprocessing.text import Tokenizer
import gc

if __name__ == "__main__":

    ## TODO: argparse the trained model path name, n_words, and n_sample

    test = unpickle_object("test_clean.pkl")
    #test = unpickle_object("test_data.pkl")
    test_corpus = " ".join(test)

    tokenizer = unpickle_object("tokenizer.pkl")
    # vocabulary size
    vocab_size = len(tokenizer.word_index) + 1

    #**** changeable variables ******
    # number of samples generated
    n_sample = 5
    # number of words generated
    n_words = 20
    #*************

    # sequence length
    seq_length = 20
    # generate new test sequences
    n_seq = 21
    test_seq = create_input_output(transcript=test_corpus, n_seq=n_seq)

    # encode the train_seq
    test_seq_encode = tokenizer.texts_to_sequences(test_seq)
    test_seq_in = np.array(test_seq_encode)[:,:-1]
    test_seq_out = np.array(test_seq_encode)[:,-1]

    #********* path name to the train model ********
    # ******** change to the correct path name *****
    model_path = "Models_GRU/GRU_2.1_weights.07-4.98.hdf5"
    #************************
    
    K.clear_session()
    model = load_model(model_path)
    model.summary()

    evaluate_model(model, tokenizer, seq_length, n_sample,
               test, test_seq_in, test_seq_out, n_words)
    

    
