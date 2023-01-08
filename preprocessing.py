import re
import math
import time
import nltk
import spacy

import numpy as np

from nltk.stem.porter import *
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

# Replace .<char> with . <char>


def replace_dot(text: str) -> str:
    regex = r'\.[\w]'
    search_result = re.search(regex, text)
    if search_result:
        return re.sub(regex, text[search_result.start():search_result.end()].replace('.', '. '), text)
    return text

# Decontracted


def decontracted(phrase: str) -> str:
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

# Punctuation removal


def punctuation_removal(text: str) -> str:
    return re.sub(r'[^\w\s]', '', text)

# Remove duplicate space


def remove_duplicate_space(text: str) -> str:
    return re.sub(r'\s+', ' ', text)

# Preprocessing


def preprocessing(text: str) -> str:
    text = text.casefold()
    text = replace_dot(text)
    # text = decontracted(text)
    # text = punctuation_removal(text)
    # text = re.sub(r'\d+', '', text)
    # text = remove_duplicate_space(text)
    return text


def alpha_numeric_removal(text: list) -> list:
    return [word for word in text if word.isalpha()]

# Stopword Removal


def stopword_removal(text: list, typed=None) -> list:
    if typed == None:
        return list()

    stop_words = stopword_mapper[typed]

    if typed == spacy_val:
        return [word for word in text if word not in stop_words]
    elif typed == nltk_val:
        return [word for word in text if word.lower() not in stop_words]

# Query Expansion


def query_expansion(text: list) -> list:
    result = []
    for word in text:
        synsets = wordnet.synsets(word)
        if synsets:
            synset = synsets[0]
            lemmas = synset.lemma_names()
            result.extend(lemmas)
    return result

# Remove Bad Words


def remove_bad_words(text: list) -> list:
    return [word for word in text if word not in list_bad_words]

# Phrase Detection


def phrase_detection(text: list) -> list:
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(' '.join(text))
    result = []
    for chunk in doc.noun_chunks:
        result.append(chunk.text)
    return result

# Stemming


def stemming(text: list) -> list:
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in text]

# Lemmatization


def lemmatization(text: list) -> list:
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in text]

# Clean


def clean(text: list, token_len=1) -> list:
    result = []
    for token in text:
        if len(token) > token_len and not token.isnumeric():
            result.append(token.lower())
    return result

# Show Wordcloud


def show_wordcloud(data, title=None):
    wordcloud = WordCloud(background_color='black',).generate(str(data))

    fig = plt.figure(1, figsize=(8, 8))
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.show()

# Negation Handling


def negation(sentence: list) -> list:
    temp = int(0)
    for i in range(len(sentence)):
        if sentence[i-1] in ['not', "n't"]:
            antonyms = []
            for syn in wordnet.synsets(sentence[i]):
                syns = wordnet.synsets(sentence[i])
                w1 = syns[0].name()
                temp = 0
                for l in syn.lemmas():
                    if l.antonyms():
                        antonyms.append(l.antonyms()[0].name())
                max_dissimilarity = 0
                for ant in antonyms:
                    syns = wordnet.synsets(ant)
                    w2 = syns[0].name()
                    syns = wordnet.synsets(sentence[i])
                    w1 = syns[0].name()
                    word1 = wordnet.synset(w1)
                    word2 = wordnet.synset(w2)
                    if isinstance(word1.wup_similarity(word2), float) or isinstance(word1.wup_similarity(word2), int):
                        temp = 1 - word1.wup_similarity(word2)
                    if temp > max_dissimilarity:
                        max_dissimilarity = temp
                        antonym_max = ant
                        sentence[i] = antonym_max
                        sentence[i-1] = ''
    while '' in sentence:
        sentence.remove('')
    return sentence

# Array to String


def array_to_string(text: list) -> str:
    return ' '.join(text)
