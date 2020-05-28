# Imports
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
import numpy as np
import os
import pickle

model_dir = 'model/'
# Tokenizers
tokenizer_5k = 'tokenizer_5k.pickle'
tokenizer_full = 'tokenizer_full.pickle'
tokenizer_full_newlines = 'tokenizer_full_newlines.pickle'
tokenizer_5k_newlines = 'tokenizer_5k_newlines.pickle'
tokenizer_songs_newlines = 'tokenizer_songs_newlines.pickle'
tokenizer_songs_newlines_100 = 'tokenizer_songs_newlines_100.pickle'
tokenizer_verse_1 = 'tokenizer_verse_newlines_1.pickle'  # Goes 5 verses each element with new lines
# Models
model_a = 'model_a.h5'  # Single LSTM(100)
model_b = 'model_b.h5'  # LSTM(150) on top of LSTM(250)
model_c = 'model_c.h5'  # LSTM(250) using songs not line by line
model_d = 'model_d.h5'  # embeddings(400) LSTM(250) using corpus line by line with no new lines
model_e = 'model_e.h5'  # LSTM (250) but go through corpus by 5 verses each time
model_f = 'model_f.h5'  # LSTM(400) verse approach

# Input sequences
input_sequence_verses_5 = 'input_sequence_verses_5.pickle'
input_sequence_songs = 'input_sequence_songs.pickle'
input_sequence_full = 'input_sequence_full.pickle'


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_stochastic_sampling(seed_text, next_words, temperature, model_name, tokenizer_name, input_seq):
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
        if output_word is '\n':  # For html
            seed_text += '<br>'
        else:
            if prev_word is '\n':
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
