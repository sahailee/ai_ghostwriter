# Imports
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import numpy as np
import os
import string
import pickle

model_dir = 'model'


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_stochastic_sampling(seed_text, next_words, temperature,
                                 model_name='model_f.h5', tokenizer_name='tokenizer_verse_newlines_1.pickle',
                                 input_seq='input_sequence_verses_5.pickle'):
    seed_text = seed_text.translate(str.maketrans('', '', string.punctuation))
    seed_text = seed_text.lower()
    tokenizer = load_tokenizer(tokenizer_name)
    input_sequences = load_tokenizer(input_seq)
    max_sequence_len = max([len(x) for x in input_sequences])

    model = load_model(os.path.join(model_dir, model_name))
    prev_word = ''
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict(token_list)[0]
        next_index = sample(predicted, temperature)
        for word, index in tokenizer.word_index.items():
            if index == next_index:
                output_word = word
                break
        if prev_word == '\n':
            seed_text += output_word
        else:
            seed_text += " " + output_word
        prev_word = output_word
    return seed_text


def load_tokenizer(name):
    # load tokenizer
    with open(os.path.join(model_dir, name), 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer
