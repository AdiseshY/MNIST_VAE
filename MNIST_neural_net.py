#build variational autoencoder and classify using a neural network 
#import standard packages
import pandas as pd 
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf 
import matplotlib.pyplot as plt 
from classes import *

data = tf.keras.datasets.mnist.load_data(path="mnist.npz")


(x_train, y_train), (x_test, y_test) = data
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

#visualize dataset 

visualize(10, 2, 5, x_train, y_train)






