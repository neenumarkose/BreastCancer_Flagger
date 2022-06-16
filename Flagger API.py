# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 08:32:02 2022

@author: neenu
"""

import numpy as np
import pickle
import pandas as pd
from flask import Flask, request
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)


@app.route('/')
def hello():
    return "Breast Cancer Prediction"


@app.route('/predict_test', methods=["POST"])
def predict_test_class():
    
    """Let's predict the breast cancer class
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: test
        in: formData
        type: file
        required: true
    responses:
        200:
            description: The output values
        
    """
    df_test=pd.read_csv(request.files.get("test"))
    prediction=classifier.predict(df_test)
    return " The Predicated Class for the TestFile is"+ str(list(prediction))


if __name__=='__main__':
    app.run()