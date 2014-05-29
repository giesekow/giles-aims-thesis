# -*- coding: utf-8 -*-
"""
Created on Fri May  9 12:39:43 2014

@author: giles
"""
import numpy as np
from sklearn.cross_validation import train_test_split
import theano
from scipy import ndimage

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
    
 
    
data_batch_1 = unpickle('real_object_data/data_batch_1')
full_data = data_batch_1['data']
full_labels = data_batch_1['labels'] 


data_batch_2 = unpickle('real_object_data/data_batch_2')
full_data = np.append(full_data,data_batch_2['data'],axis=0)
full_labels = np.append(full_labels,data_batch_2['labels'],axis=0)


data_batch_3 = unpickle('real_object_data/data_batch_3')
full_data = np.append(full_data,data_batch_3['data'],axis=0)
full_labels = np.append(full_labels,data_batch_3['labels'],axis=0)

data_batch_4 = unpickle('real_object_data/data_batch_4')
full_data = np.append(full_data,data_batch_4['data'],axis=0)
full_labels = np.append(full_labels,data_batch_4['labels'],axis=0)

data_batch_5 = unpickle('real_object_data/data_batch_5')
full_data = np.append(full_data,data_batch_5['data'],axis=0)
full_labels = np.append(full_labels,data_batch_5['labels'],axis=0)


test_batch = unpickle('real_object_data/test_batch')

rng = np.random.RandomState(1234)


def get_datasets():
    train_set_x,test_set_x,train_set_y,test_set_y = train_test_split(full_data,full_labels,train_size=0.8,random_state=rng)
    train_set_x,valid_set_x,train_set_y,valid_set_y = train_test_split(full_data,full_labels,train_size=0.8,random_state=rng)
    
    train_set_x = get_fft_transformed_data(train_set_x)/255
    valid_set_x = get_fft_transformed_data(valid_set_x)/255
    test_set_x = get_fft_transformed_data(test_set_x)/255
    
    train_set_x = theano.shared(np.asarray(train_set_x,dtype=theano.config.floatX),borrow=True)
    train_set_y = theano.shared(np.asarray(train_set_y,dtype=theano.config.floatX),borrow=True) 
    
    valid_set_x = theano.shared(np.asarray(valid_set_x,dtype=theano.config.floatX),borrow=True)
    valid_set_y = theano.shared(np.asarray(valid_set_y,dtype=theano.config.floatX),borrow=True) 
    
    test_set_x = theano.shared(np.asarray(test_set_x,dtype=theano.config.floatX),borrow=True)
    test_set_y = theano.shared(np.asarray(test_set_y,dtype=theano.config.floatX),borrow=True) 
    
    
    return [[train_set_x,train_set_y],[valid_set_x,valid_set_y],[test_set_x,test_set_y]]

def get_training_data():
    return get_fft_transformed_data(full_data),full_labels
    
def get_test_data():
    return get_fft_transformed_data(test_batch['data']),test_batch['labels']
    
def get_fft_transformed_data(data):
    dshape = np.shape(data)
    new_data = data.reshape(dshape[0],3,32,32)
    new_data = np.transpose(new_data,(0,2,3,1))
    new_data = np.mean(new_data,axis=3)
    kernel = np.array([[-1, -1, -1],[-1,  8, -1],[-1, -1, -1]])
    
    for i in range(dshape[0]):
        new_data[i] = ndimage.convolve(new_data[i], kernel)
        new_data[i] = abs(new_data[i])/np.max(abs(new_data[i]))
       
    return new_data.reshape(dshape[0],32*32)