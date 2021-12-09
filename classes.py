import pandas as pd 
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf 
import matplotlib.pyplot as plt

#visualize data

def visualize(num, num_row, num_col, x_train, y_train): 
    images = x_train[:num]
    labels = y_train[:num]

    fig, axes = plt.subplots(num_row, num_col, figsize = (1.5*num_row, 1.5*num_col))
    for i in range(num): 
        ax = axes[i//num_col, i%num_col]
        ax.imshow(images[i], cmap = 'gray')
        ax.set_title('Label: {}'.format(labels[i]))
    plt.tight_layout()
    plt.show()
