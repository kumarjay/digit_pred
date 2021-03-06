# -*- coding: utf-8 -*-

from flask import Flask, render_template, request
#from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
import re
import base64
import sys
import os
import scipy as sc
import cv2


sys.path.append(os.path.abspath('./model'))
from load import *

global model, graph

app= Flask(__name__)


model, graph= init()

print(graph.as_default())

#def convertImage(imgData1):
#    imgstr= re.search(r'base64,(.*)',imgData1).group(1)
#    
#    with open('output.png','wb') as output:
#        output.write(imgstr.decode('base64'))

def convertImage(imgData1): 
    imgstr = re.search(b'base64,(.*)',imgData1).group(1) 
    #print(imgstr) 
    with open('output.png','wb') as output: 
        output.write(base64.b64decode(imgstr))
#
@app.route('/')
def index():
    return render_template('index.html')
#
#@app.route('/predict/', methods=['GET','POST'])
#def predict():
#    imgData= request.get_data()
#    convertImage(imgData)
#    print('debug 1')
#    
#    x= imread('output.png', mode='L')
#    x= np.invert(x)
#    x= imresize(x, (28,28))
#    x= x.reshape(1,28,28,1)
#    print('debug 2')
#    
#    with graph.as_default():
#        out= model.predict(x)
#        print(out)
#        print(np.argmax(out, axis=1))
#        print('debug 3')
#        response= np.array_str(np.argmax(out, axis=1))
#        return response
    

@app.route('/predict/', methods=['GET','POST'])
def predict():
    imgData= request.get_data()
    convertImage(imgData)
    print('debug 1')
    
    x= cv2.imread('output.png',0)
    x= np.invert(x)
    x= cv2.resize(x, (28,28))
    x= x.reshape(1,28,28,1)
    print('debug 2')
    
    with graph.as_default():
        out= model.predict(x)
        print(out)
        print(np.argmax(out, axis=1))
        print('debug 3')
        response= np.array_str(np.argmax(out, axis=1))
        return response
    
    


if __name__ == "__main__":
	#decide what port to run the app in
    #	port = int(os.environ.get('PORT', 5000))
	#run the app locally on the givn port
    #	app.run(host='0.0.0.0', port=port)
	#optional if we want to run in debugging mode
	app.run(debug=True)
    
    
    
    

