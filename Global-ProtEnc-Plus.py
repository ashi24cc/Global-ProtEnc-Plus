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
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt
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

def final_model(filename, segSize, nonOL):
    max_seq_len = segSize - chunkSize + 1
    overlap = 30

    model_path = '/content/gdrive/My Drive/Multi-Attn/Molecular Function/Simple+Rank+Attention(4add)/64.model_'+str(nonOL)+'_'+ str(segSize) +'.h5' 
    model = load_model(model_path, custom_objects={'categorical_crossentropy': categorical_crossentropy})
    print(model.summary())

    print('Extracting features based on LSTM model...... ')
    dataframe2 = pd.read_csv(filename, header=None)
    dataset2 = dataframe2.values
    X_test = dataset2[:,0]
    Y_test = dataset2[:,1:len(dataset2[0])]

    c_p = []
    for tag, row in enumerate(X_test):
        pos = math.ceil(len(row) / overlap)
        if(pos < math.ceil(segSize/ overlap)):
            pos = math.ceil(segSize/ overlap)
        segment = [ ]
        for itr in range(pos - math.ceil(segSize/overlap) + 1):
            init = itr * overlap
            segment.append(row[init : init + segSize])
        seg_nGram = nGram(segment, chunkSize, dict_Prop)
        test_seg = sequence.pad_sequences(seg_nGram, maxlen=max_seq_len)
        preds = model.predict(test_seg)
        c_p.append(cls_predict(preds))
    c_p = np.array(c_p)

    del model
    return c_p, Y_test

def create_nn_model(dim):
    n_model = Sequential()
    n_model.add(Dense(dim, input_dim = dim, kernel_initializer='normal', activation='relu'))
    n_model.add(Dense(dim, kernel_initializer='normal', activation='sigmoid'))
    n_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return n_model

# CREATING DICTIONARY
chunkSize = 4
dict_Prop = dictionary(chunkSize)
train_path = "/content/gdrive/My Drive/Multi-Attn/Molecular Function/trainData.csv"  # Set the path

X_train_new1, Y_train_new = final_model(train_path, 80, 40)
model11 = create_nn_model(Y_train_new[0].shape[0])
print(model11.summary())
early_stopping_monitor = EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 1)
model11.fit(X_train_new1, Y_train_new.astype(None),
           callbacks = [early_stopping_monitor],
           validation_split = 0.1,
           epochs = 1000,
           batch_size = 150,
           verbose = True)

X_train_new2, _ = final_model(train_path, 100, 50)
model12 = create_nn_model(Y_train_new[0].shape[0])
print(model12.summary())
early_stopping_monitor = EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 1)
model12.fit(X_train_new2, Y_train_new.astype(None),
           callbacks = [early_stopping_monitor],
           validation_split = 0.1,
           epochs = 1000,
           batch_size = 150,
           verbose = True)

X_train_new3, _ = final_model(train_path, 120, 60)
model13 = create_nn_model(Y_train_new[0].shape[0])
print(model13.summary())
early_stopping_monitor = EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 1)
model13.fit(X_train_new3, Y_train_new.astype(None),
           callbacks = [early_stopping_monitor],
           validation_split = 0.1,
           epochs = 1000,
           batch_size = 150,
           verbose = True)

# Testing
def test_fun(file):
    X_test_new1, Y_test_new = final_model(file, 80, 40)
    X_test_new2, _ = final_model(file, 100, 50)
    X_test_new3, _ = final_model(file, 120, 60)

    print(X_test_new1.shape, Y_test_new.shape)
    print(X_test_new2.shape, Y_test_new.shape)
    print(X_test_new3.shape, Y_test_new.shape)
    Y_test_new = np.array(Y_test_new).astype(None)

    fmax, tmax = 0.0, 0.0
    precisions, recalls = [], []
    for t in range(1, 101, 1):
        test_preds1 = model11.predict(X_test_new1)
        test_preds2 = model12.predict(X_test_new2)
        test_preds3 = model13.predict(X_test_new3)
        test_preds = (test_preds1 + test_preds2 + test_preds3) / 3

        threshold = t / 100.0
        print("THRESHOLD IS =====> ", threshold)
        test_preds[test_preds>=threshold] = int(1)
        test_preds[test_preds<threshold] = int(0)

        rec = recall(Y_test_new, test_preds)
        pre = precision(Y_test_new, test_preds)
        if math.isnan(pre):
            pre = 0.0
        recalls.append(rec)
        precisions.append(pre)

        f1 = f_score(Y_test_new, test_preds)*100
        f = 2 * pre * rec / (pre + rec)
        print('Recall: {0}'.format(rec*100), '     Precision: {0}'.format(pre*100),
              '     F1-score1: {0}'.format(f*100), '      F1-score2: {0}'.format(f1))

        if fmax < f:
            fmax = f
            tmax = threshold
    
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    aupr = np.trapz(precisions, recalls)
    print(f'AUPR: {aupr:0.3f}')

    plt.figure()
    plt.plot(recalls, precisions, color='darkorange', lw=2, label=f'AUPR curve (area = {aupr:0.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Area Under the Precision-Recall curve')
    plt.legend(loc="upper right")
    plt.savefig(f'aupr.pdf')

    return tmax

th_set = test_fun("/content/gdrive/My Drive/Multi-Attn/Molecular Function/testData.csv")
print("Best Threshold: ", th_set)
