import json
import numpy as np
import tensorflow
import tensorflow.keras
import oversampling
import os
import os.path

path_prelim = '../'
test_fname= path_prelim + 'test.json'

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
        
    smote = data['smote']
    
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
    
    x_flat = x_test.flatten().tolist()
    y_flat = y_test.flatten().tolist()
    x_shape = list(x_test.shape)
    y_shape = list(y_test.shape)
    
    output = {}
    output['tensors'] = []
    output['tensors'].append({'content': x_flat, 'dims': x_shape})
    output['tensors'].append({'content': y_flat, 'dims': y_shape})
    
    with open('../test_out.json', 'w') as out_f:
        json.dump(output, out_f)