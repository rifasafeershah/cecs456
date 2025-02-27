import numpy as np 
from helper import *
'''
Rifa Safeer Shah
Homework3_sp21: logistic regression classifier (Coding Assignment 2)
'''

'''
I would personally avoid using the linear model which does not have the third order testing. The 
third order regression provides an improved scores. With the first order test cases the max iteration case and 
learning rate test case scored poorly for both the testing and training data. I would test with both first and third 
order before making a final deliverable to a customer.
'''

def logistic_regression(data, label, max_iter, learning_rate):
    '''
    The logistic regression classifier function.
    Args:
    data: train data with shape (1561, 3), which means 1561 samples and
          each sample has 3 features.(1, symmetry, average internsity)
    label: train data's label with shape (1561,1).
           1 for digit number 1 and -1 for digit number 5.
    max_iter: max iteration numbers
    learning_rate: learning rate for weight update
    Returns:
        w: the seperater with shape (3, 1). You must initilize it with w = np.zeros((d,1))
    '''
    
    w = np.zeros((data.shape[1], 1))
    
    for i in range(max_iter):
            
        k = np.zeros((data.shape[1],1))
        
        for (xi, yi) in zip(data, label):
                
            xi = xi.reshape((data.shape[1],1))
            
            k = k + yi * xi * sigmoid(-yi * w.T.dot(xi)) #use sigmoid funct
            
        k /= len(label)
        
        w = w + k * learning_rate
        
    return w

def thirdorder(data):
    '''
    This function is used for a 3rd order polynomial transform of the data.
    Args:
    data: input data with shape (:, 3) the first dimension represents
          total samples (training: 1561; testing: 424) and the
          second dimesion represents total features.
    Return:
        result: A numpy array format new data with shape (:,10), which using
        a 3rd order polynomial transformation to extend the feature numbers
        from 3 to 10.
        The first dimension represents total samples (training: 1561; testing: 424)
        and the second dimesion represents total features.
    '''
    polydegree = 3
    
    result = np.ones((data.shape[0], 1))
    
    data1 = data[:, 0]
    
    data2 = data[:, 1]
    
    for i in range(1, polydegree + 1):
            
        for j in range(0, i + 1):
                
            column = (data1 ** (i - j)) * (data2 ** j)
            
            result = np.append(result, column.reshape(column.shape[0], 1), axis = 1)
            
    return result


def accuracy(x, y, w):
    '''
    This function is used to compute accuracy of a logsitic regression model.
    
    Args:
    x: input data with shape (n, d), where n represents total data samples and d represents
        total feature numbers of a certain data sample.
    y: corresponding label of x with shape(n, 1), where n represents total data samples.
    w: the seperator learnt from logistic regression function with shape (d, 1),
        where d represents total feature numbers of a certain data sample.
    Return 
        accuracy: total percents of correctly classified samples. Set the threshold as 0.5,
        which means, if the predicted probability > 0.5, classify as 1; Otherwise, classify as -1.
    '''
    activate = sigmoid(np.dot(x,w)) #use sigmoid funct to activate
    
    predict = np.where(activate > 0.5, 1, -1) #predicted probabilty (classification as 1 or -1)
    
    return np.mean(predict == y.reshape((y.shape[0], 1)))

#sigmoid function to calculate the sigmoid activation
def sigmoid(s):
    return 1 / (1 + np.exp(-s))
