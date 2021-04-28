import numpy as np
from flask import Flask, request, render_template
import pickle

from fastai.vision.all import *
from fastai.torch_basics import *
from fastai.data.all import *
from fastai.text.core import *
from fastai.text.data import TextDataLoaders
from fastai.text.learner import language_model_learner
from fastai.text import *
from fastai.text.all import *

import os
import gc
application = Flask(__name__)

import pandas as pd

app = Flask(__name__)

import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


file_id = '1THaxJhzeElP5A_efR8juHefbLDp5cEMO'
destination = 'model.pkl'

download_file_from_google_drive(file_id, destination)
gc.collect()

model = load_learner('model.pkl')
labels = model.dls.vocab[1]

@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    prediction = model.predict(text)
    preds_df = pd.concat([pd.DataFrame(labels),pd.DataFrame(prediction[2])],axis=1)
    preds_df.columns = ["Label", "Probability"]
    result = preds_df.sort_values('Probability',ascending=False)
    return result.to_json()