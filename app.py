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

@app.route('/predict', methods=['POST'])
def predict():
    
    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    
    output=round(prediction[0],2)
    
    return render_template('index.html', prediction_text='Salary should be ${}'.format(output))

if __name__=='__main__':
    app.run(debug=True)
