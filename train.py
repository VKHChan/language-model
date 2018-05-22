'''
Train model
- unpickle encoded training and validation data
- unpickle tokenzier

- generate model
- train model
'''

from helper import *
from models import *
import random

if __name__ == "__main__":
    # TODO: argparse algorithm and network hyperparameters

    train_seq_encode = unpickle_object("train_seq_encode.pkl")
    valid_seq_encode = unpickle_object("valid_seq_encode.pkl")

    tokenizer = unpickle_object("tokenizer.pkl")
    print(type(tokenizer))

    # use a subset of training and validation data to train smaller models for now
    sub_size_train = int(len(train_seq_encode) * 0.05)
    sub_size_valid = int(sub_size_train *0.14)
    print("training sub size:", sub_size_train)
    print("validation sub size:", sub_size_valid)

    # use a subset of training and validation data to train smaller models for now
    #train_sub = np.array(random.sample(train_seq_encode, sub_size_train))
    #valid_sub = np.array(random.sample(valid_seq_encode, sub_size_valid))
    train_sub = np.array(train_seq_encode[:sub_size_train])
    valid_sub = np.array(valid_seq_encode[:sub_size_valid])
    print(train_sub.shape, valid_sub.shape)

    n_seq = train_sub.shape[1]
    vocab_size = len(tokenizer.word_index) + 1

    batch_size = 128
    hidden_size = 256
    num_epoch = 100
    emb_size = 50
    num_layer = 2
    drop_out = 0.2
    seq_length = n_seq-1

    model = build_GRU(vocab_size=vocab_size, seq_length=seq_length, emb_size=emb_size, 
                        num_layers=num_layer, drop_out=drop_out, hidde_size=hidden_size)

    file_path = "GRU_weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_loss',
                            verbose=1, save_best_only=True, mode='min')
    earlystop = EarlyStopping(monitor='val_loss', mode='min', patience=10)
    callback_list = [checkpoint, earlystop]

    model.fit_generator(my_generator(train_sub, n_seq, vocab_size, tokenizer, batch_size),
                        validation_data = my_generator(valid_sub, n_seq, vocab_size, tokenizer, batch_size),
                        validation_steps = int(len(valid_sub)/batch_size),
                        steps_per_epoch = (len(train_sub)/batch_size),
                        epochs=num_epoch, callbacks=callback_list)
