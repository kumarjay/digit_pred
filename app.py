from __future__ import print_function
import numpy as np
from flask import Flask, request, jsonify, render_template
import os
#from scipy.misc import imsave, imread, imresize
#import pickle

#for importing our keras model
#import keras.models
#for regular expressions, saves time dealing with string data
import re


app= Flask(__name__)
#model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/', methods=['GET','POST'])
def predict():
    imgData= request.get_data()
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
