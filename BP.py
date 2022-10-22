import math
import numpy as np
from keras.datasets import mnist
import csv
import time
import os 
from os import listdir
from os.path import isfile, join
import cv2

(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_path = "Your path"
scale_train = len(train_X)
scale_valid = len(train_y)

def sigmoid(x):
    if x >= 0:
        return 1.0/(1.0 + np.exp(-x)) 
    else:
        return np.exp(x)/(1.0 + np.exp(x)) 

class BP(object):
    
    def __init__(self, input_size, fc_layers = [512,64,10]):
        self.weights = []
        in_channels = input_size
        for i in range(len(fc_layers)):
            self.weights.append(np.random.normal(size=(in_channels, fc_layers[i])))
            in_channels = fc_layers[i]
    
    def Forward_pass(self, x, weights):
        res = np.dot(x, weights)
        res = 1.0/(1.0 + np.exp(-res)) 
        return res
   
    def Backward_pass(self, lr, sigma, layer_output, w):
        sigma_ex = sigma[None, :]
        output_ex = layer_output[:, None]
        w += np.matmul(output_ex, sigma_ex) * lr
        return w
        
    def train(self, data, targets, epoches, lr, idx=1):
        acc = 0
        for epoch in range(epoches):
            print("執行開始".center(10//2, '-'))
            start = time.perf_counter()
            true_num = 0
            idx = 1
            for r, t in zip(data, targets):
                ratio = (idx / scale_train)
                movedBar = '*' * int(10*ratio)
                residualBar = '.' * int(10*(1 - ratio))
                dur = time.perf_counter() - start
                print("\r{:3.0f}%[{}->{}]{:.2f}s acc:{:.2f}% ".format(ratio*100, movedBar, residualBar, dur, (true_num/idx)*100), end='')
                r= r/255
                r = r.flatten()
                target = np.zeros(10)
                target[t] = 1
                fc_output = []
                fc_in = r
                for i in range(len(self.weights)):
                    fc_output.append(self.Forward_pass(fc_in, self.weights[i]))
                    fc_in = fc_output[-1]
                tmp = [idx, t, fc_output[-1].argmax(), 1 if t == fc_output[-1].argmax() else 0]
                idx+=1
                loss = target - fc_output[-1]
                if(t == fc_output[-1].argmax()):
                    true_num+=1
                sigma = []
                coefficient = loss
                for i in range(len(fc_output)):
                    sigma.append(coefficient*fc_output[-1-i]*(1-fc_output[-1-i]))
                    coefficient = np.dot(self.weights[-1-i], sigma[-1])
                coefficient_a = r
                for i in range(len(self.weights)):
                    self.weights[i] = self.Backward_pass(lr, sigma[-1-i], coefficient_a, self.weights[i])
                    coefficient_a = fc_output[i]
                acc = (true_num/idx)*100
            print("\n"+"執行結束".center(10//2, '-'))
            print("")
            break
        return acc
    
    def predict(self, data, targets, idx=1):
        print("執行開始".center(10//2, '-'))
        start = time.perf_counter()
        true_num = 0
        idx = 1
        acc = 0
        for r, t in zip(data, targets):
            ratio = (idx / scale_valid)
            movedBar = '*' * int(10*ratio)
            residualBar = '.' * int(10*(1 - ratio))
            dur = time.perf_counter() - start
            print("\r{:3.0f}%[{}->{}]{:.2f}s acc:{:.2f}% ".format(ratio*100, movedBar, residualBar, dur, (true_num/idx)*100), end='')
            r = r.flatten()
            r = r/255
            fc_output = []
            fc_in = r
            for i in range(len(self.weights)):
                fc_output.append(self.Forward_pass(fc_in, self.weights[i]))
                fc_in = fc_output[-1]
            tmp = [idx, t, fc_output[-1].argmax(), 1 if t == fc_output[-1].argmax() else 0]
            if t == fc_output[-1].argmax(): 
                true_num+=1
            idx+=1
            acc = (true_num/idx)*100
        print("\n"+"執行結束".center(10//2, '-'))
        print("")
        return acc
    
    def test(self, data, idx=1):
        print("執行開始".center(10//2, '-'))
        for r in data:
            arr = cv2.imread(r, cv2.IMREAD_GRAYSCALE)
            arr = arr.flatten()
            arr = arr/255
            fc_output = []
            fc_in = arr
            for i in range(len(self.weights)):
                fc_output.append(self.Forward_pass(fc_in, self.weights[i]))
                fc_in = fc_output[-1]
            idx+=1
        print("\n"+"執行結束".center(10//2, '-'))
        print("")
        return
    
    def print_weights(self):
        for i in range(len(self.weights)):
            print(self.weights[i].shape)

            
def main():
    for i in range (3):
    train_X = np.array(train_X)
    train_y = np.array(train_y)
    indices = np.arange(train_X.shape[0])
    np.random.shuffle(indices)
    train_X = train_X[indices]
    train_y = train_y[indices]
    print(train_X.shape)
    print(train_y.shape)
    model.train(data=train_X, targets=train_y, epoches=1, lr=0.1)
    model.predict(data=train_X, targets=train_y)
