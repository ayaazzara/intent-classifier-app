import os
import streamlit as st
import pickle

from nltk.tokenize import word_tokenize

from preprocessing import *
from tfidf import TFIDF
from mlr import MultinomialLogisticRegression

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


class Models():
    def __init__(self, tfidf, mlr):
        self.tfidf = tfidf
        self.mlr = mlr


def input_text():
    text = st.text_input('Enter some text')
    if st.button('Predict'):
        return text


def main():
    st.title('Intent Classification Demo')
    models = load_models()
    text = input_text()
    if text:
        pred = predict(text, models)
        st.write(f'The query intent is **{label_kelas[pred[0]]}**')


def load_models():
    file_x_train = os.path.join(model_dir, 'X_train.sav')
    file_x_test = os.path.join(model_dir, 'X_test.sav')
    file_y_train = os.path.join(model_dir, 'Y_train.sav')
    file_y_test = os.path.join(model_dir, 'Y_test.sav')
    file_tfidf = os.path.join(model_dir, 'tfidf.sav')
    model_MLR = os.path.join(model_dir, 'BestMLRModel.sav')

    # X, Y Data Train
    X_train = pickle.load(open(file_x_train, 'rb'))
    Y_train = pickle.load(open(file_y_train, 'rb'))

    # X, Y Data Test
    X_test = pickle.load(open(file_x_test, 'rb'))
    Y_test = pickle.load(open(file_y_test, 'rb'))

    # TF-IDF
    tfidf = pickle.load(open(file_tfidf, 'rb'))

    # MLR Model
    Best_MLR = pickle.load(open(model_MLR, 'rb'))

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
