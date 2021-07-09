from flask import Flask, request, render_template
import pickle
import pandas as pd
from collections import OrderedDict
import json

from fastai.vision.all import *


import nltk
nltk.download('wordnet')
nltk.download('punkt')

from utils.cleantext import clean_text
from utils.gdrive import download_file_from_google_drive

application = Flask(__name__)

app = Flask(__name__)

model_id = '1lMm8pTVJubGbbrjt1BiHSvEiC6TPZRd-'
destination = 'model.pkl'

download_file_from_google_drive(model_id, destination)

model = load_learner(destination)
labels = model.dls.vocab[1]

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/model')
def get_model_id():
    return model_id

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    threshold = data.get('threshold', 0)
    precision = data.get('precision', 3)
    if threshold > 1 or threshold < 0:
        threshold = 0
    if precision < 0:
        precision = 3
    cleaned = clean_text(text)
    prediction = model.predict(cleaned)
    preds_df = pd.concat([pd.DataFrame(labels),
                         pd.DataFrame(prediction[2])], axis=1)
    preds_df.columns = ['Label', 'Probability']
    filtered_df = preds_df[preds_df['Probability'] >= threshold]
    filtered_df['Probability'] = filtered_df['Probability'
            ].round(decimals=precision)
    preds_dict = dict(zip(filtered_df.Label, filtered_df.Probability))
    preds_dict_sorted = sorted(preds_dict.items(), key=lambda x: x[1],
                               reverse=True)
    return json.dumps({
        'input': text,
        'cleaned': cleaned,
        'threshold': threshold,
        'precision': precision,
        'prediction': OrderedDict(preds_dict_sorted),
        })