
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, SimpleRNN, GRU
from keras.layers import Flatten
from keras.layers import Activation, Dropout, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

#from helper import *

'''
Codes used to build different RNN models:
RNN modles include - SimpleRNN, LSTM, GNU

Input:
num_layers = number of hidden layers
num_hidden_neurons = number of hidden neurons
drop_out = drop out for each layer
vocab_size = vocabolary size
seq_length = length of input sequence

Output:
the required RNN model
'''
def build_simpleRNN(vocab_size, seq_length, emb_size = 50, num_layers = 2,
                    drop_out = 0.5, hidde_size = 100):
    # define model
    K.clear_session()
    model = Sequential()
    model.add(Embedding(vocab_size, emb_size, input_length=seq_length))
    for i in range(num_layers):
        model.add(SimpleRNN(hidde_size, return_sequences=(i != num_layers-1)))
        model.add(Dropout(drop_out))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))

    print(model.summary())

    # compile model
    model.compile(loss = 'categorical_crossentropy',
             optimizer = 'adam',
             metrics = ['accuracy'])

    return model

def build_LSTM(vocab_size, seq_length, emb_size = 50, num_layers = 2,
                    drop_out = 0.5, hidde_size = 100):
    # define model
    K.clear_session()
    model = Sequential()
    model.add(Embedding(vocab_size, emb_size, input_length=seq_length))
    for i in range(num_layers):
        model.add(LSTM(hidde_size, return_sequences=(i != num_layers-1)))
        model.add(Dropout(drop_out))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))

    print(model.summary())

    # compile model
    model.compile(loss = 'categorical_crossentropy',
             optimizer = 'adam',
             metrics = ['accuracy'])

    return model

def build_GRU(vocab_size, seq_length, emb_size = 50, num_layers = 2,
                    drop_out = 0.5, hidde_size = 100):
    # define model
    K.clear_session()
    model = Sequential()
    model.add(Embedding(vocab_size, emb_size, input_length=seq_length))
    for i in range(num_layers):
        model.add(GRU(hidde_size, return_sequences=(i != num_layers-1)))
        model.add(Dropout(drop_out))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))

    print(model.summary())

    # compile model
    model.compile(loss = 'categorical_crossentropy',
             optimizer = 'adam',
             metrics = ['accuracy'])

    return model

def build_GRU_glove(vocab_size, seq_length, embedding_matrix, emb_size = 50, num_layers = 2,
                    drop_out = 0.5, hidde_size = 100):
    # define model
    K.clear_session()
    model = Sequential()
    model.add(Embedding(vocab_size, emb_size, weights=[embedding_matrix],
                        input_length=seq_length, trainable=False))
    for i in range(num_layers):
        model.add(GRU(hidde_size, return_sequences=(i != num_layers-1)))
        model.add(Dropout(drop_out))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))

    print(model.summary())

    # compile model
    model.compile(loss = 'categorical_crossentropy',
             optimizer = 'adam',
             metrics = ['accuracy'])

    return model

'''
Generator function for fit_generator()
'''
def my_generator(data_seq_encoded, n_seq, vocab_size, tokenizer, batch_size):
    batch_X = np.zeros((batch_size, n_seq-1), dtype=int)
    batch_Y = np.zeros((batch_size, vocab_size))
    while True:
        for i in range(batch_size):
            index = np.random.choice(len(data_seq_encoded),1)
            batch_X[i] = data_seq_encoded[index,:-1]
            temp_y = data_seq_encoded[index,-1]
            batch_Y[i] = to_categorical(temp_y, num_classes=vocab_size)
        yield batch_X, batch_Y

'''
Function to generate text given seed_text and model
'''
def generate_text(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text
    print(in_text)
    print(tokenizer.texts_to_sequences([in_text])[0])
    index_word = {v: k for k, v in tokenizer.word_index.items()}
    for _ in range(n_words):
        # encode text with tokenizer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequence to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        #print(encoded)
        y = model.predict_classes(encoded, verbose=0)

        # map predicted word index to word
        out_word = index_word.get(y[0])

        # append to input
        in_text += " " + out_word
        #in_text.append(out_word)

        result.append(out_word)
    return " ".join(result)
