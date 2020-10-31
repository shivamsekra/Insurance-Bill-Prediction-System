# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 22:24:43 2020

@author: Admin
"""


from flask import Flask,request, url_for, redirect, render_template, jsonify,config
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model =  pickle.load(open(r'C:\Users\Admin\Desktop\Insurance_data_end_to_end\model.pkl','rb'))
cols = ['age', 'bmi', 'children','sex',  'smoker', 'region']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final], columns = cols)
    prediction = model.predict(data_unseen)
    prediction = int(prediction[0])
    return render_template('home.html',pred='Expected Bill will be {}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = model.predict(data_unseen)
    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)