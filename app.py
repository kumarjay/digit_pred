from __future__ import print_function
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
import base64
import scipy as sc
import sys
import re
#from keras imprt models

sys.path.append(os.path.abspath('./model'))
from load import *

app= Flask(__name__)

global model, graph
model, graph= init()


model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
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

    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
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

    output=round(prediction[0],2)

    return render_template('index.html', prediction_text='Salary should be ${}'.format(output))

if __name__=='__main__':
    app.run(debug=True)
    #port = int(os.environ.get('PORT', 5000))
    #run the app locally on the givn port
    #app.run(host='0.0.0.0', port=port)
