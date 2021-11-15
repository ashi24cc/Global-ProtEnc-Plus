def final_model(filename):
    print('Extracting features based on LSTM model...... ')
    dataframe2 = pd.read_csv(filename, header=None)
    dataset2 = dataframe2.values
    overlap = 30
    X_test = dataset2[:,0]
    Y_test = dataset2[:,1:len(dataset2[0])]
    c_p = []
    for tag, row in enumerate(X_test):
        pos = math.ceil(len(row) / overlap)
        if(pos < math.ceil(segmentSize/ overlap)):
            pos = math.ceil(segmentSize/ overlap)
        segment = [ ]
        for itr in range(pos - math.ceil(segmentSize/overlap) + 1):
            init = itr * overlap
            segment.append(row[init : init + segmentSize])
        seg_nGram = nGram(segment, chunkSize, dict_Prop)
        test_seg = sequence.pad_sequences(seg_nGram, maxlen=max_seq_len)
        preds = model.predict(test_seg)
        c_p.append(cls_predict(preds))
    c_p = np.array(c_p)
    return c_p, Y_test

# Creates a HDF5 file 'my_model.h5'
#model_path = '/content/gdrive/My Drive/Multi-Attn/Molecular Function/Simple+Rank+Attention(4add)/64.model_'+str(nonOL)+'_'+ SEG +'.h5'
#model.save(model_path)
#del model  
#model = load_model(model_path, custom_objects={'categorical_crossentropy': categorical_crossentropy})

# Training
path = "/content/gdrive/My Drive/Multi-Attn/Molecular Function/trainData.csv"  # Set the path to training folder.
X_train_new, Y_train_new = final_model(path)

# Training model 2
model1 = create_nn_model(Y_train_new[0].shape[0])
print(model1.summary())
early_stopping_monitor = EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 1)
model1.fit(X_train_new, Y_train_new.astype(None),
           callbacks = [early_stopping_monitor],
           validation_split = 0.1,
           epochs = 1000,
           batch_size = 150,
           verbose = True)
