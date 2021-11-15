import numpy as np
import math
%tensorflow_version 1.x
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, CuDNNGRU, Bidirectional, Input, Dropout, Add
from keras.layers import Flatten, Activation, RepeatVector, Permute, multiply, Lambda
from keras import backend as K
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt
from 
np.random.seed(7)

def epsilon():
    _EPSILON = 1e-7
    return _EPSILON

def _to_tensor(x, dtype):
    return tf.convert_to_tensor(x, dtype=dtype)

def categorical_crossentropy(target, output, from_logits=False):
    if not from_logits:
        output /= tf.reduce_sum(output, len(output.get_shape()) - 1, True)
        _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
        return - tf.reduce_sum(target * tf.log(output), len(output.get_shape()) - 1)
    else:
        return tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=output)

def cls_predict(pred, normalize=True, sample_weight=None):
    pred1 = pred[0]
    pred2 = pred[1]
    y_pred1 = (pred2)**(1-pred1)
    y_pred2 = (pred1)**(1-pred2)
    y_pred = y_pred1 + y_pred2
    s_mean = np.mean(y_pred, axis=0)
    m = max(s_mean)
    s_mean = (s_mean/m)
    return(list(s_mean))

def dictionary(chunk_size):
    dataframe = pd.read_csv("/content/gdrive/My Drive/Multi-Attn/Molecular Function/trainData.csv", header=None)
    dataset = dataframe.values
    seq_dataset = dataset[:,0]
    print('Creating Dictionary:')
    dict = {}
    j = 0
    for row in seq_dataset:
        for i in range(len(row) - chunk_size + 1):
            key = row[i:i + chunk_size]
            if key not in dict:
                dict[key] = j
                j = j + 1
    del dataframe, dataset, seq_dataset
    return(dict)

def nGram(dataset, chunk_size, dictI):
    dict1 = list()
    for j, row in enumerate(dataset):
        string = row
        dict2 = list()
        for i in range(len(string) - chunk_size + 1):
            try:
                dict2.append(dictI[string[i:i + chunk_size]])
            except:
                None
        dict1.append(dict2)   
    return(dict1)

def create_rec_model1(top_words, seq_len, o_dim):
    embedding_vecor_length = 64

    _input = Input(shape=[seq_len])
    embdd = Embedding(top_words, embedding_vecor_length, input_length = seq_len)(_input)
    drop1 = Dropout(0.4)(embdd)
    activations = Bidirectional(CuDNNGRU(100, return_sequences=True))(drop1)

    # compute importance for each step
    attention1 = Dense(1, activation='tanh')(activations)
    attention1 = Flatten()(attention1)
    attention1 = Activation('softmax')(attention1)
    
    attention2 = Dense(1, activation='tanh')(activations)
    attention2 = Flatten()(attention2)
    attention2 = Activation('softmax')(attention2)

    attention3 = Dense(1, activation='tanh')(activations)
    attention3 = Flatten()(attention3)
    attention3 = Activation('softmax')(attention3)

    attention4 = Dense(1, activation='tanh')(activations)
    attention4 = Flatten()(attention4)
    attention4 = Activation('softmax')(attention4)
    
    attention = Add()([attention1,attention2,attention3,attention4])
    attention = RepeatVector(200)(attention)
    attention = Permute([2, 1])(attention)
    
    sent_representation = multiply([activations, attention])
    sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)

    drop2 = Dropout(0.5)(sent_representation)

    den1 = Dense(o_dim, kernel_initializer='normal', activation='softmax')(drop2)
    den2 = Dense(o_dim, kernel_initializer='normal', activation='sigmoid')(drop2)

    r_model = Model(inputs = [_input], outputs = [den1,den2])
    r_model.compile(loss=[categorical_crossentropy,'binary_crossentropy'], loss_weights=[0.30, 1.0],
                    optimizer='adam', metrics=['accuracy'])
    return r_model

def create_nn_model(dim):
    n_model = Sequential()
    n_model.add(Dense(dim, input_dim = dim, kernel_initializer='normal', activation='relu'))
    n_model.add(Dense(dim, kernel_initializer='normal', activation='sigmoid'))
    n_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return n_model

# CREATING DICTIONARY
chunkSize = 4
dict_Prop = dictionary(chunkSize)

# Preparing For Training
segmentSize = 100
nonOL = 50
SEG = str(segmentSize)
main_fun(segmentSize, nonOL)                                       # Create segments
dataframe = pd.read_csv("trainData_Seg.csv", header=None)
dataset = dataframe.values

X = dataset[:,0]
Y = dataset[:,1:len(dataset[0])].astype(None)
nb_of_cls = len(Y[0])
del dataframe, dataset

#Split the dataset
x_train, x_validate, y_train, y_validate = train_test_split(X, Y, test_size = 0.1, random_state = 42)
del X, Y

#CREATING N-GRAM
x_train = nGram(x_train, chunkSize, dict_Prop)
x_validate = nGram(x_validate, chunkSize, dict_Prop)

# truncate and pad input sequences
max_seq_len = segmentSize - chunkSize + 1
x_train = sequence.pad_sequences(x_train, maxlen=max_seq_len)
x_validate = sequence.pad_sequences(x_validate, maxlen=max_seq_len)

# Create & Compile the model
model = create_rec_model1(len(dict_Prop), max_seq_len, nb_of_cls)
print(model.summary())
early_stopping_monitor1 = EarlyStopping(monitor = 'val_loss', patience = 2, verbose = 1)
history = model.fit(x_train, [y_train, y_train],
          validation_data = (x_validate, [y_validate, y_validate]),
          epochs = 1000,
          batch_size = 150,
          callbacks=[early_stopping_monitor1],
          verbose=1)
