from scikitclassifiers import *
import numpy as np
from sklearn.cross_validation import train_test_split
import time


if __name__ == '__main__':
    print "loading data...."
    data = np.load("digit_data/digit_train_set.npy")
    Y = data[:,0]
    X = data[:,1:]/255
    
    rng = np.random.RandomState(1234)
    rng1 = np.random.RandomState(3214)
    d_train_set_x,d_test_set_x,d_train_set_y,d_test_set_y = train_test_split(X,Y,test_size=8500,random_state=rng)
    d_train_set_x,d_valid_set_x,d_train_set_y,d_valid_set_y = train_test_split(d_train_set_x,d_train_set_y,test_size=8500,random_state=rng1)    
 
    
    print "training size:",np.shape(d_train_set_x)
    print "validation size:",np.shape(d_valid_set_x)
    print "test size:",np.shape(d_test_set_x)    
    
    print "Performing feature extraction"        
    maxp_train_set_x = extract_feature_by_Max_pooling(d_train_set_x,(2,2))
    maxp_valid_set_x = extract_feature_by_Max_pooling(d_valid_set_x,(2,2))
    maxp_test_set_x = extract_feature_by_Max_pooling(d_test_set_x,(2,2))
    
    np.save("digit_pred/test_set_labels.npy",d_test_set_y)
    
    print "Performing training and predicting on normal data set"
    eval_times = []
    
    print "Knn....."
    start_time = time.clock()
    norm_knn_pred = knn_train_and_predict(d_train_set_x,d_train_set_y,d_test_set_x,d_test_set_y)        
    end_time = time.clock()
    eval_times.append(end_time-start_time)
    np.save("digit_pred/normal_knn.npy",norm_knn_pred)  
    
    
    print "logistic Regression..." 
    start_time = time.clock()
    norm_logistic_pred = logistic_train_and_predict(d_train_set_x,d_train_set_y,d_test_set_x,d_test_set_y)
    end_time = time.clock()    
    np.save("digit_pred/normal_logistic.npy",norm_logistic_pred)  
    eval_times.append(end_time-start_time)
    
    
    print "scikit DBN...."
    start_time =time.clock()
    norm_dbn_pred = dbn_train_and_predict(d_train_set_x,d_train_set_y,d_test_set_x,d_test_set_y)
    end_time = time.clock()    
    np.save("r_obj_pred/normal_dbn.npy",norm_dbn_pred)
    eval_times.append(end_time-start_time)
    
    
    eval_times = np.array(eval_times)
    np.save("digit_pred/exec_times_norm.npy",eval_times)
    
    
  
    print "Performing training and predicting on pca data"
    eval_times = []
     
    print "Knn..."
    start_time = time.clock()
    pca_knn_pred = pca_knn_train_and_predict(d_train_set_x,d_train_set_y,d_test_set_x,d_test_set_y)
    end_time =time.clock()
    np.save("digit_pred/pca_knn.npy",pca_knn_pred)
    eval_times.append(end_time-start_time)
    
    
    print "logistic regression..." 
    start_time = time.clock()
    pca_logistic_pred = pca_logistic_train_and_predict(d_train_set_x,d_train_set_y,d_test_set_x,d_test_set_y)
    end_time = time.clock()    
    np.save("digit_pred/pca_logistic.npy",pca_logistic_pred)
    eval_times.append(end_time-start_time)
    
    eval_times = np.array(eval_times)
    np.save("digit_pred/exec_times_pca.npy",eval_times)
    
        
    
    print "Performing training and prediction on rbm data"
    eval_times = []
    
    print "Knn..."
    start_time = time.clock()
    rbm_knn_pred = rbm_knn_train_and_predict(d_train_set_x,d_train_set_y,d_test_set_x,d_test_set_y)
    end_time = time.clock()    
    np.save("digit_pred/rbm_knn.npy",rbm_knn_pred)
    eval_times.append(end_time-start_time)
    
    print "logistic regression..." 
    start_time = time.clock()
    rbm_logistic_pred = rbm_logistic_train_and_predict(d_train_set_x,d_train_set_y,d_test_set_x,d_test_set_y)
    end_time = time.clock()    
    np.save("digit_pred/rbm_logistic.npy",rbm_logistic_pred)
    eval_times.append(end_time-start_time)
    
    eval_times = np.array(eval_times)
    np.save("digit_pred/exec_times_rbm.npy",eval_times)
    
    print "Performing training and prediction on max pooled data"
    eval_times = []
    
    print "Knn..."
    start_time = time.clock()
    maxp_knn_pred = knn_train_and_predict(maxp_train_set_x,d_train_set_y,maxp_test_set_x,d_test_set_y)
    end_time = time.clock()   
    np.save("digit_pred/maxp_knn.npy",maxp_knn_pred)
    eval_times.append(end_time-start_time)
    
    print "logistic regression..."
    start_time = time.clock()
    maxp_logistic_pred = logistic_train_and_predict(maxp_train_set_x,d_train_set_y,maxp_test_set_x,d_test_set_y)
    end_time = time.clock()    
    np.save("digit_pred/maxp_logistic.npy",maxp_logistic_pred)
    eval_times.append(end_time-start_time)
    
    
    print "scikit DBN..."  
    start_time = time.clock()
    maxp_dbn_pred = dbn_train_and_predict(maxp_train_set_x,d_train_set_y,maxp_test_set_x,d_test_set_y)
    end_time = time.clock()    
    np.save("digit_pred/maxp_dbn.npy",maxp_dbn_pred)
    eval_times.append(end_time-start_time)
    
    eval_times = np.array(eval_times)
    np.save("digit_pred/exec_times_maxp.npy",eval_times)
    
        
    
    print "Multilayer perceptron..."
    eval_times = []
    print "Normal data"
    start_time = time.clock()    
    norm_perceptron_pred = mlp_train_and_predict(d_train_set_x,d_train_set_y,d_valid_set_x,d_valid_set_y,d_test_set_x,d_test_set_y)
    end_time = time.clock()    
    np.save("digit_pred/normal_perceptron.npy",norm_perceptron_pred)    
    eval_times.append(end_time-start_time)
    
    print "Multilayer Perceptron..." 
    print "Max pooled"
    start_time = time.clock()
    maxp_perceptron_pred = mlp_train_and_predict(maxp_train_set_x,d_train_set_y,maxp_valid_set_x,d_valid_set_y,maxp_test_set_x,d_test_set_y)
    end_time = time.clock()    
    np.save("digit_pred/maxp_perceptron.npy",maxp_perceptron_pred)
    eval_times.append(end_time-start_time)
    
    eval_times = np.array(eval_times)
    np.save("digit_pred/exec_times_mlp.npy",eval_times)
    
    
    print "Convolutional MLP..."
    start_time = time.clock()    
    norm_cov_mlp_pred = conv_mlp_train_and_predict(maxp_train_set_x,d_train_set_y,maxp_valid_set_x,d_valid_set_y,maxp_test_set_x,d_test_set_y,rng)
    end_time = time.clock()    
    np.save("digit_pred/normal_conv_mlp.npy",norm_cov_mlp_pred)
    eval_times.append(end_time-start_time)
    
    eval_times = np.array(eval_times)
    np.save("digit_pred/exec_times_conv_mlp.npy",eval_times)
    
    print "DBN Using theano..."
    start_time = time.clock()
    norm_t_dbn_pred = t_dbn_train_and_predict(maxp_train_set_x,d_train_set_y,maxp_valid_set_x,d_valid_set_y,d_test_set_x,maxp_test_set_y)
    end_time = time.clock()    
    np.save("digit_pred/normal_theano_dbn.npy",norm_t_dbn_pred)
    eval_times.append(end_time-start_time)
    
    eval_times = np.array(eval_times)
    np.save("digit_pred/exec_times_t_dbn.npy",eval_times)
   