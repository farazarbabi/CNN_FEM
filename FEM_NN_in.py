
# coding: utf-8

"""
This is an independant research project designed and implemented by Faraz Arbabi, Ph.D.

This module includs creating deep learning layers and models.
All rights reserved by Faraz Arbabi, Ph.D.
Created on Thu Feb  8 21:01:57 2018
"""

#%reset

import numpy as np
import glob 

import sklearn.model_selection as skms

import keras.layers as kl
import keras.models as km

######################################################################

def get_data():

        allfiles = glob.glob('surfaceCSVs\\*.csv')

        num_datapoints = 1071

        in_data = np.zeros([len(allfiles) * num_datapoints, 3])
        out_data = np.zeros(len(allfiles) * num_datapoints)


        for i, datafile in enumerate(allfiles):

                force = int(datafile.split('\\')[1].split('-')[0]) / float(50000)
               # print(force)
                # put the force in
                in_data[i * num_datapoints:
                        (i + 1) * num_datapoints, 0] = \
                                np.array([force] * num_datapoints)
                
                data = np.loadtxt(fname = datafile, delimiter = ' ')
                
                # get the x data
                in_data[i * num_datapoints:
                        (i + 1) * num_datapoints, 1] = data[:, 0]
                
                # get the y data
                in_data[i * num_datapoints:
                        (i + 1) * num_datapoints, 2] = data[:, 1]
                
                # get sigma
                out_data[i * num_datapoints:
                         (i + 1) * num_datapoints] = data[:, 3] / float(50000)
        

        return skms.train_test_split(in_data, out_data,
                                     test_size = 0.1)
        

######################################################################

def build_model(numnodes):

        model = km.Sequential()

        model.add(kl.Dense(numnodes, input_dim = 3,
                           activation = 'sigmoid'))

        model.add(kl.Dense(numnodes, activation = 'sigmoid'))

        model.add(kl.Dense(1, activation = 'linear'))

        return model


######################################################################

def abs_pred(y_true, y_pred):
        return abs((y_true - y_pred) / y_true) < 0.1


######################################################################

def build_model2(numnodes):

        input_position = kl.Input(shape = (2,))
        input_force = kl.Input(shape = (1,))

        x_position = kl.Dense(numnodes,
                              activation = 'tanh')(input_position)

        x_force = kl.Dense(numnodes, activation = 'relu')(input_force)

        x = kl.concatenate(inputs = [x_position, x_force])
        #x = kl.multiply(inputs = [x_position, x_force])
        
        x = kl.Dense(numnodes, activation = 'tanh')(x)

        x = kl.Dense(numnodes, activation = 'tanh')(x)

        x = kl.Dense(1, activation = 'linear')(x)
        
        model = km.Model(inputs = [input_position, input_force],
                         outputs = x)

        return model

