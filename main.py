import os
import streamlit as st
import pickle
import pandas as pd

from nltk.tokenize import word_tokenize

from preprocessing import *
from tfidf import TFIDF
from mlr import MultinomialLogisticRegression

from googletrans import Translator

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Global variables
label_kelas = {
    1: "Inform",
    2: "Question",
    3: "Directive",
    4: "Commissive"
}

# current path
current_path = os.path.dirname(os.path.abspath(__file__))

model_dir = os.path.join(current_path, 'models')

translator = Translator()


class Models():
    def __init__(self, tfidf, mlr):
        self.tfidf = tfidf
        self.mlr = mlr


def input_text():
    text = st.text_input('Enter some text')
    if st.button('Predict'):
        return text


def translate(text: str) -> str:
    try:
        return translator.translate(text, dest='en', src='auto').text
    except:
        return text


def detect_lang(text: str) -> str:
    try:
        return translator.detect(text).lang
    except:
        return 'en'


def main():
    st.title('Intent Classification Demo')
    st.subheader(
        'Aplikasi ini digunakan untuk klasifikasi intent menggunakan Multinomial Logistic Regression. \n\n Terdapat 4 macam intent yang dapat dideteksi, antara lain: \n 1. Inform: menyampaikan atau memberi informasi. \n 2. Question: bertanya atau menyampaikan pertanyaan. \n 3. Directive: memberikan perintah. \n 4. Commissive: memberikan janji atau komitmen. \n\n Langkah-langkah untuk melakukan prediksi: \n 1. Masukkan text yang ingin diprediksi \n 2. Klik tombol "Predict" \n 3. Hasil prediksi akan ditampilkan di bawah tombol "Predict"')
    models = load_models()
    text = input_text()

    if text is None:
        return

    lang_type = detect_lang(text)
    if len(text) == 0:
        return

    if lang_type != 'en':
        text = translate(text)

    pred = predict(text, models)
    st.write(f'The query language is **{lang_type}**')
    st.write(f'The query translated is **{text}**')
    st.write(f'The query intent is **{label_kelas[pred[0]]}**')


def load_models():
    file_x_train = os.path.join(model_dir, 'X_train.sav')
    file_x_test = os.path.join(model_dir, 'X_test.sav')
    file_y_train = os.path.join(model_dir, 'Y_train.sav')
    file_y_test = os.path.join(model_dir, 'Y_test.sav')
    file_tfidf = os.path.join(model_dir, 'tfidf.sav')
    model_MLR = os.path.join(model_dir, 'BestMLRModel.sav')

    # X, Y Data Train
    X_train = pd.read_pickle(open(file_x_train, 'rb'))
    Y_train = pd.read_pickle(open(file_y_train, 'rb'))

    # X, Y Data Test
    X_test = pd.read_pickle(open(file_x_test, 'rb'))
    Y_test = pd.read_pickle(open(file_y_test, 'rb'))

    # TF-IDF
    tfidf = pd.read_pickle(open(file_tfidf, 'rb'))

    # MLR Model
    Best_MLR = pd.read_pickle(open(model_MLR, 'rb'))

    return Models(tfidf, Best_MLR)


def predict(text: str, model: Models):
    tfidf = model.tfidf
    mlr = model.mlr

    text = preprocessing(text)
    text = word_tokenize(text)
    text = negation(text)
    text = lemmatization(text)
    text = clean(text)
    text = array_to_string(text)
    text = tfidf.transform([text])

    pred_text = mlr.predict(text.toarray())

    return pred_text


if __name__ == '__main__':
    main()
