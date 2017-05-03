
import scipy.io
import numpy as np
import sklearn.utils as sku
import sklearn.preprocessing as prep

def getData():
    base = scipy.io.loadmat('base_ready_data.mat')['base_ready_data']
    left = scipy.io.loadmat('left_ready_data.mat')['left_ready_data']
    right = scipy.io.loadmat('right_ready_data.mat')['right_ready_data']

    # concatenate all of them
    whole_data = np.concatenate((base,right,left),axis=2)
    # reshape it and then transpose 
    whole_data = np.reshape(whole_data,(57,128*3)).transpose()

    # Make the correct answers
    base_label = np.concatenate((np.ones(128),np.zeros(256)))
    left_label = np.concatenate((np.zeros(128), np.ones(128),np.zeros(128)))
    right_label = np.concatenate((np.zeros(256),np.ones(128)))

    whole_label = np.concatenate((base_label,left_label,right_label))
    whole_label = np.reshape(whole_label,(3,128*3)).transpose()
    # Normalize the data
    prep.normalize(whole_data)
    return whole_data,whole_label

def shuffleData(data, label):
    random_data,random_label = sku.shuffle(data,label)
    # print(random_data.shape)
    # print(random_label.shape)

    
    return random_data,random_label