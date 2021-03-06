# -*- coding: utf-8 -*-

import numpy as np
import keras
#from scipy.misc import imread, imresize, imshow
import tensorflow as tf

def init():
    json_file= open('model.json', 'r')
    loaded_model_json= json_file.read()
    json_file.close()
    loaded_model= keras.models.model_from_json(loaded_model_json)
    
    loaded_model.load_weights('model.h5')
    print('Loaded model from disk')
    
    loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    graph= tf.compat.v1.get_default_graph()
    
    return loaded_model, graph

