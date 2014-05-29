# -*- coding: utf-8 -*-
"""
Created on Thu May 15 20:17:34 2014

@author: giles
"""

from sklearn import linear_model,metrics
from sklearn.neural_network import BernoulliRBM
from sklearn.neighbors import KNeighborsClassifier
from nolearn.dbn import DBN
from perceptron import test_mlp
from sklearn.decomposition import PCA
import theano.tensor as T
import numpy as np
import theano
from theano.tensor.signal import downsample
from convolutional_mlp import evaluate_lenet5
from DBN import test_DBN
from sklearn.pipeline import Pipeline


def logistic_train_and_predict(train_set_x,train_set_y,test_set_x,test_set_y):
    classifier = linear_model.LogisticRegression(C=6000)
    classifier.fit(train_set_x,train_set_y)
    PRED = classifier.predict(test_set_x)
    return PRED
    


def knn_train_and_predict(train_set_x,train_set_y,test_set_x,test_set_y):
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(train_set_x,train_set_y)
    PRED = classifier.predict(test_set_x)
    return PRED
    

def dbn_train_and_predict(train_set_x,train_set_y,test_set_x,test_set_y):
    classifier = DBN(epochs=200,learn_rates=0.01)
    classifier.fit(train_set_x,train_set_y)
    PRED = classifier.predict(test_set_x)
    return PRED

def mlp_train_and_predict(train_set_x,train_set_y,valid_set_x,valid_set_y,test_set_x,test_set_y):
    PRED = test_mlp(np.copy(train_set_x),np.copy(train_set_y),np.copy(valid_set_x),np.copy(valid_set_y),np.copy(test_set_x),np.copy(test_set_y))
    return PRED
    
def conv_mlp_train_and_predict(train_set_x,train_set_y,valid_set_x,valid_set_y,test_set_x,test_set_y,rng,isrbg=False):
    PRED = evaluate_lenet5(np.copy(train_set_x),np.copy(train_set_y),np.copy(valid_set_x),np.copy(valid_set_y),np.copy(test_set_x),np.copy(test_set_y),rng,isrbg=isrbg)
    return PRED
    
def t_dbn_train_and_predict(train_set_x,train_set_y,valid_set_x,valid_set_y,test_set_x,test_set_y):
    PRED = test_DBN(np.copy(train_set_x),np.copy(train_set_y),np.copy(valid_set_x),np.copy(valid_set_y),np.copy(test_set_x),np.copy(test_set_y))
    return PRED
    

def rbm_logistic_train_and_predict(train_set_x,train_set_y,test_set_x,test_set_y):
    logistic = linear_model.LogisticRegression(C=6000)
    rbm = BernoulliRBM(random_state=0, verbose=True)
    rbm.learning_rate = 0.06
    rbm.n_iter = 20
    rbm.n_components = 100
    classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
    classifier.fit(train_set_x,train_set_y)
    PRED = classifier.predict(test_set_x)
    return PRED
    


def rbm_knn_train_and_predict(train_set_x,train_set_y,test_set_x,test_set_y):
    knn = KNeighborsClassifier(n_neighbors=5)
    rbm = BernoulliRBM(random_state=0, verbose=True)
    rbm.learning_rate = 0.06
    rbm.n_iter = 20
    rbm.n_components = 100
    classifier = Pipeline(steps=[('rbm', rbm), ('knn', knn)])
    classifier.fit(train_set_x,train_set_y)
    PRED = classifier.predict(test_set_x)
    return PRED
    

def rbm_dbn_train_and_predict(train_set_x,train_set_y,test_set_x,test_set_y):
    dbn = DBN(epochs=200,learn_rates=0.01)
    rbm = BernoulliRBM(random_state=0, verbose=True)
    rbm.learning_rate = 0.06
    rbm.n_iter = 20
    rbm.n_components = 100
    classifier = Pipeline(steps=[('rbm', rbm), ('dbn', dbn)])
    classifier.fit(train_set_x,train_set_y)
    PRED = classifier.predict(test_set_x)
    return PRED   
    
def pca_logistic_train_and_predict(train_set_x,train_set_y,test_set_x,test_set_y):
    logistic = linear_model.LogisticRegression(C=6000)
    pca = PCA(n_components=0.9)
    classifier = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
    classifier.fit(train_set_x,train_set_y)
    PRED = classifier.predict(test_set_x)
    return PRED
    


def pca_knn_train_and_predict(train_set_x,train_set_y,test_set_x,test_set_y):
    knn = KNeighborsClassifier(n_neighbors=5)
    pca = PCA(n_components=0.9)
    classifier = Pipeline(steps=[('pca', pca), ('knn', knn)])
    classifier.fit(train_set_x,train_set_y)
    PRED = classifier.predict(test_set_x)
    return PRED
    

def pca_dbn_train_and_predict(train_set_x,train_set_y,test_set_x,test_set_y):
    dbn = DBN(epochs=200,learn_rates=0.01)
    pca = PCA(n_components=0.9)
    classifier = Pipeline(steps=[('pca', pca), ('dbn', dbn)])
    classifier.fit(train_set_x,train_set_y)
    PRED = classifier.predict(test_set_x)
    return PRED   
    
    
    
    
    
    
    
def extract_feature_by_Max_pooling(dataset,maxpool_shape,dim=3):
    if dim ==4:
        tshp = np.shape(dataset)
        dataset = dataset.reshape(tshp[0],1,int(np.sqrt(tshp[1]/3.0)),int(np.sqrt(tshp[1]/3.0)),3)
        dataset = np.mean(dataset,axis=4)
        dataset = dataset.reshape(tshp[0],tshp[1]/3)
        
        
    input = T.dtensor3('input')

    pool_out = downsample.max_pool_2d(input, maxpool_shape, ignore_border=True)
    f = theano.function([input],pool_out)
    r,c = np.shape(dataset)
    new_data = dataset.reshape(r,int(np.sqrt(c)),int(np.sqrt(c)))
    
    pooled_data = f(new_data)
    
    shp = np.shape(pooled_data) 
    print shp
    
    pooled_data = pooled_data.reshape(r,shp[1]*shp[2])
    return pooled_data
    
    
