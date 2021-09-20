import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import tokenizer_from_json

import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')

from flask import Flask, request, render_template


app = Flask(__name__, template_folder='templates', static_folder='statics')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        news = request.form.get('link')
        prediction = create_test_input(news)
        return prediction


def create_test_input(message):
    temp = ("".join(message.split(' ')))
    if len(message.split(' ')) < 4 and temp.isalnum() and not temp.isdigit():
        return "Enter a Headline with more words"
    elif temp.isnumeric():
        return "Invalid headline"

    model = load_model('models/latest_model.h5')
    max_length = 31

    f = open('tokenizer/latest_tokenizer.json')
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)
    f.close()
    corpus = []
    review = re.sub('[^a-zA-Z]', ' ', message)
    review = review.split()
    #    review = [word for word in review]

    review = [word for word in review if not word in stopwords.words('english')]

    corpus.append(review)

    sequences = tokenizer.texts_to_sequences(corpus)

    if len(sequences[0]) < 3:
        return "Enter a Headline with more words"

    data = pad_sequences(sequences, maxlen=max_length)
    X_final = np.array(data)
    prediction = (model.predict(X_final) > 0.7).astype("int32")
    if prediction[0][0] == 1:
        return "Genuine news"
    else:
        return "Fake news"


if __name__ == '__main__':
    app.run(debug=False)
