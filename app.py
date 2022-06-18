import joblib
from flask import Flask, jsonify, request, render_template
import nltk
nltk.download('wordnet')
nltk.download('punkt')
import pandas
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
import numpy as np
import re
import flask
app = Flask(__name__)
#clf = joblib.load('modelrfc.pkl')
clf = joblib.load('classification.pkl')
count_vect = joblib.load('tfidvect.pkl')

@app.route('/')
def index():
    #return Flask.render_template('index.html')
    return render_template('index.html')


def pre_processing(text):
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    text = re.sub('[0-9]+','num',text)
    word_list = nltk.word_tokenize(text)
    word_list =  [lemmatizer.lemmatize(item) for item in word_list]
    return ' '.join(word_list)


@app.route('/predict', methods=['POST'])
def predict():
    to_predict_list = request.form.to_dict()
    review_text = pre_processing(to_predict_list['review_text'])

    print(count_vect.transform([review_text]))
    pred = clf.predict(count_vect.transform([review_text]))
    prob = clf.predict_proba(count_vect.transform([review_text]))

    print('Predict # {}'.format(pred))
    # pr =  1
    #comment
    if prob[0][0] >= 0.5:
        prediction = "Positive"
        # pr = prob[0][0]
    else:
        prediction = "Negative"
        # pr = prob[0][0]

    # sanity check to filter out non questions.
    if not re.search("(?i)(what|which|who|where|why|when|how|whose|\?)", to_predict_list['review_text']):
        prediction = "Negative"
        # prob = prob*0

    return render_template('predict.html', prediction=prediction, prob=np.round(prob[0][0], 3) * 100)



if __name__ == '__main__':
    #clf = joblib.load('quora_model.pkl')
    #count_vect = joblib.load('quora_vectorizer.pkl')
    app.run(debug=True)
    #app.run(host='localhost', port=8081)