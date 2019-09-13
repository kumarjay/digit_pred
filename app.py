# -*- coding: utf-8 -*-

from flask import Flask, render_template, request
from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
import re

import sys
import os

sys.path.append(os.path.abspath('./model'))
from load import *

global model, graph

model, graph= init()

def convertImage(imgData1):
    imgstr= re.search(r'base64,(.*)',imgData1).group(1)
    
    with open('output.png','wb') as output:
        output.write(imgstr.decode('base64'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    imgData= request.getData()
    convertImage(imgData)
    print('debug 1')
    
    x= imread('output.png', mode='L')
    x= np.invert(x)
    x= imresize(x, (28,28))
    x= x.reshape(1,28,28,1)
    print('debug 2')
    
    with graph.as_default():
        out= model.predict(x)
        print(out)
        print(np.argmax(out, axis=1))
        print('debug 3')
        response= np.array_str(np.argmax(out, axis=1))
        return response
    
if __name__=='__main__':
    app.run(debug=True)
