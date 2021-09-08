from model_definitions import ModelDefinitions
import json
import numpy as np
import tensorflow
import tensorflow.keras
import oversampling
import os
import os.path

DELETE_AFTER_USE = True

funcs = [ModelDefinitions.callMFCC_CNN, ModelDefinitions.callSpectrogram_CNN, ModelDefinitions.callMelSpectrogram_CNN,
         ModelDefinitions.callMFCC_RNN, ModelDefinitions.callSpectrogram_RNN, ModelDefinitions.callMelSpectrogram_RNN,
         ModelDefinitions.callMFCC_LSTM, ModelDefinitions.callSpectrogram_LSTM, ModelDefinitions.callMelSpectrogram_LSTM]

path_prelim = '../'
train_fname= path_prelim + 'train.json'
test_fname= path_prelim + 'test.json'

# Train part

if(os.path.isfile(train_fname)):
    tensor_list = []
    file = open(train_fname)
    data = json.load(file)
    
    for i in data['tensors']:
        content = i['content']
        dims = i['dims']
        dims.append(1)
        np.asarray(content)
        content = np.reshape(content, dims)
        tensor_list.append(content)
        
    smote = data['smote']
    model_id = data['model']
    epoch_count = data['epoch']
    
    file.close()
    
    if((model_id - 1) % 3 == 0):
        ModelDefinitions.mfcc_length = np.shape(tensor_list[0])[1];
        ModelDefinitions.mfcc_width = np.shape(tensor_list[0])[2];
    elif((model_id - 1) % 3 == 1):
        ModelDefinitions.spectro_length = np.shape(tensor_list[0])[1];
        ModelDefinitions.spectro_width = np.shape(tensor_list[0])[2];
    else:
        ModelDefinitions.mel_spectro_length = np.shape(tensor_list[0])[1];
        ModelDefinitions.mel_spectro_width = np.shape(tensor_list[0])[2];
    
    model = funcs[model_id - 1]();
    
    working_size = np.zeros(len(tensor_list[0]))
    failing_size = np.ones(len(tensor_list[1]))
    failed_size  = 2 * np.ones(len(tensor_list[2]))
    working_size = tensorflow.keras.utils.to_categorical(working_size, 3)
    failing_size = tensorflow.keras.utils.to_categorical(failing_size, 3)
    failed_size = tensorflow.keras.utils.to_categorical(failed_size, 3)
    
    x_train = np.concatenate((tensor_list[0], tensor_list[1], tensor_list[2]), axis=0)
    y_train = np.concatenate((working_size, failing_size, failed_size), axis=0)
    
    if(smote):
        x_train, y_train = oversampling.oversampling(x_train, y_train)

# Test part 

if(os.path.isfile(test_fname)):
    tensor_list_test = []
    file_test = open(test_fname)
    data = json.load(file_test)
    
    for i in data['tensors']:
        content = i['content']
        dims = i['dims']
        dims.append(1)
        np.asarray(content)
        content = np.reshape(content, dims)
        tensor_list_test.append(content)
    
    file_test.close()
    
    working_test_size = np.zeros(len(tensor_list_test[0]))
    failing_test_size = np.ones(len(tensor_list_test[1]))
    failed_test_size  = 2 * np.ones(len(tensor_list_test[2]))
    working_test_size = tensorflow.keras.utils.to_categorical(working_test_size, 3)
    failing_test_size = tensorflow.keras.utils.to_categorical(failing_test_size, 3)
    failed_test_size = tensorflow.keras.utils.to_categorical(failed_test_size, 3)
    
    x_test = np.concatenate((tensor_list_test[0], tensor_list_test[1], tensor_list_test[2]), axis=0)
    y_test = np.concatenate((working_test_size, failing_test_size, failed_test_size), axis=0)

    if(smote):
        x_test, y_test = oversampling.oversampling(x_test, y_test)

if(os.path.isfile(train_fname)):
    if(model_id > 6):
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    if(os.path.isfile(test_fname)):
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], x_test.shape[2]))
        model.fit(x_train, y_train, epochs=epoch_count, validation_data=(x_test, y_test))
    else:
        model.fit(x_train, y_train, epochs=epoch_count)
    md_path = '../models/model_' + str(model_id)
    model.save(md_path, save_format='tf')
    
    write_to = open(md_path + "/.MODELID", "w")
    write_to.write(str(model_id))
    write_to.close()

if(DELETE_AFTER_USE):
    if(os.path.isfile(train_fname)):
        os.remove(train_fname)
    if(os.path.isfile(test_fname)):
        os.remove(test_fname)