from __future__ import print_function
import numpy as np
from flask import Flask, request, jsonify, render_template
import os
import base64
import scipy as sc
#from scipy.misc import imsave, imread, imresize
#import pickle
import sys
#for importing our keras model
#from keras import models
#for regular expressions, saves time dealing with string data
import re


sys.path.append(os.path.abspath('./model'))
from load import *

global model, graph


app= Flask(__name__)
#model=pickle.load(open('model.pkl','rb'))

model, graph= init()

@app.route('/')
def home():
    return render_template('index.html')

def convertImage(imgData1): 
    imgstr = re.search(b'base64,(.*)',imgData1).group(1) 
    #print(imgstr) 
    with open('output.png','wb') as output: 
        output.write(base64.b64decode(imgstr))
        
#from scipy.misc import imsave, imread, imresize

@app.route('/predict/', methods=['GET','POST'])
def predict():
    imgData= request.get_data()
    convertImage(imgData)
    print('debug 1')
    
    x= sc.misc.imread('output.png', mode='L')
    x= np.invert(x)
    x= sc.misc.imresize(x, (28,28))
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
