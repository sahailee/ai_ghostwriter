# Imports
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Conv1D, MaxPooling1D, GRU
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow.keras as keras
import os
import string
import pickle

# This is the file I have been running to train the model and preprocess data.
# Its extremely messy because there was a lot of experimenting I did. All the variables with the prefix
# 'model_' are a model I trained and they are all either made up of different layers or I processed the data in
# a different way.


# # Old variables ignore
# # Models trained on first 5k of corpus
# model_single_5k = 'old/single_lstm_adjusted_5k.h5'  # Single LSTM 200
# model_dual_5k = 'old/dual_lstm_adjusted_5k.h5'  # Dual LSTM 250 each
# model_single_bi_5k = 'old/single_bi_lstm_adjusted_5k.h5'  # single bi lstm 200 i think

# # Models trained on Full corpus
# model_single_full = 'old/single_lstm_adjusted_full.h5'  # Single LSTM 250

# Model that trains over song by song instead of line by line
# model_song_single_100 = 'old/single_song_lstm_adjusted_100.h5'  # Single LSTM 250. Training over a 100 songs
# model_t = 'old/single_song_lstm_100.h5'

model_dir = 'DrakeAi'
# Tokenizers: These 4 were used when each line in the corpus has an input sequence
tokenizer_5k = 'tokenizer_5k.pickle'  # Only contains the tokens for words in the first 5000 lines in the corpus. New lines not included
tokenizer_full = 'tokenizer_full.pickle'  # Contains tokens for every word in the corpus. New lines not included
tokenizer_full_newlines = 'tokenizer_full_newlines.pickle'  # Full corpus includes new lines
tokenizer_5k_newlines = 'tokenizer_5k_newlines.pickle'  # First 5k lines, includes new lines
# These tokenizers were used when I decided to split the corpus in different ways.
tokenizer_songs_newlines = 'tokenizer_songs_newlines.pickle'  # Each song is an input sequence, using all songs
tokenizer_songs_newlines_100 = 'tokenizer_songs_newlines_100.pickle'  # Only using the first 100 songs.
tokenizer_verse_1 = 'tokenizer_verse_newlines_1.pickle'  # Every 5 verses is an input sequence
# Models
model_a = 'model_a.h5'  # Single LSTM(100)
model_b = 'model_b.h5'  # LSTM(150) on top of LSTM(250)
model_c = 'model_c.h5'  # LSTM(250) using songs not line by line
model_d = 'model_d.h5'  # embeddings(400) LSTM(250) using corpus line by line with no new lines
model_e = 'model_e.h5'  # LSTM (250) but go through corpus by 5 verses each time
model_f = 'model_f.h5'  # LSTM(400) verse approach
model_g = 'model_g.h5'
model_h = 'model_h.h5'  # 1 embedding(50) 2 LSTM(100) 2 Dense(100)
model_h_full = 'model_h_full.h5'
model_g_full = 'model_g_full.h5'
songs = []  # Store the corpus where each song is stored as its own element

# Input sequences
input_sequence_verses_5 = 'input_sequence_verses_5.pickle'
input_sequence_songs = 'input_sequence_songs.pickle'
input_sequence_full = 'input_sequence_full.pickle'


# Samples from a distribution, and distribution changes based on temperature.
# This is from Francois Chollet's Deep Learning With Python Book
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate_stochastic_sampling(seed_text, next_words, temperature, model, tokenizer, max_sequence_len):
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
        if prev_word is '\n':
            seed_text += output_word
        else:
            seed_text += " " + output_word
        prev_word = output_word
    print(seed_text)


def generate(seed_text, next_words, model, tokenizer, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        #     preditced = np.argmax(model.predict(token_list), axis=-1)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    print(seed_text)


def clean_string(data):
    new_data = ''
    for char in data:
        if char > 'z' and char != '[' and char != ']':
            continue
        elif char == '-':
            new_data += ' '
            continue
        elif char == ',':
            new_data += ' ' + char
            continue
        elif char in '.?!\":':
            # new_data += ' ' + char
            continue
        elif char < '0' and char > ' ':
            continue
        else:
            new_data += char
    return new_data


#  I used this for joining all the lyrics together in one file for the gpt2 model. I do not use it for anything else
#  here
def read_and_save_corpus():
    drake_dir = 'DrakeAi/lyrics/corpus'
    base_dir = 'DrakeAi/other_artists'
    dir_list = {'carti': 'Playboi Carti', 'frank': 'Frank Ocean',
                'jayz': 'JAY Z', 'kanye': 'Kanye West', 'kendrick': 'Kendrick Lamar',
                'rihanna': 'Rihanna', 'travis': 'Travis Scott', 'weeknd': 'The Weeknd'}
    data = ''
    for folder in dir_list:
        curr = os.path.join(base_dir, folder)
        for file in os.listdir(curr):
            if file.endswith(".txt"):
                data += 'Song: Artist: ' + dir_list[folder] + '\n'
                data += open(os.path.join(curr, file), encoding="utf8").read()
                data = data.replace("â€…", " ")
                data = data.replace("âŸ", " ")
                data = data.replace("â€™", "'")
                data = data.replace("â€”", "-")
                data = data.replace("â€¦", "...")
                data = data.replace("â€˜", "'")
                data = data.replace("Ã¡", "a")
                data = data.replace("Ã±", "n")
                data = data.replace("Ã©", "é")
                data = data.replace("a³", "o")
                data = data.replace("a¨", "e")
                data = data.replace("’", "'")
                data = data.replace("—", "-")
                data = data.replace("…", "...")
                data = data.replace("”", '"')
                data = data.replace("“", '"')
                data = data.replace("Ã", "a")
                data = data.replace("–", "-")
                data = data.replace("‘", "'")
                data = data.replace(" ", "")
                data += '\n\nTitle: ' + file[:-4]
                data += '\n\n'
        for ch in data:
            if ord(ch) > 128 and ch not in ",-éóúèà":
                print(ch)
    text_file = open("DrakeAi/all_artists_title_bottom.txt", "w",
                     encoding="utf8")
    text_file.write(data)
    text_file.close()
    # text_file = open("DrakeAi/whole_corpus.txt", "r", encoding="utf8").read()
    # for ch in text_file:
    #     if ord(ch) > 128 and ch is not',' and ch is not '-':
    #         print(ch)


# Read all the lyrics from all the files in the directory and put them in a single array.
# If with_new_line variable is True then, include new lines as separate words otherwise get rid of new line characters
# If verses is -1, then each sequence appended to corpus will be one line of lyrics.
# If verses is greater than 0, then each sequence appended to corpus will be verses lines of lyrics.
# Also each full song lyrics is appended to the song[] array
def read_text_from_corpus(with_new_line=False, verses=-1):
    # Getting all of the texts in the corpus
    base_dir = 'DrakeAi/lyrics/corpus'
    corpus = []
    verse_array = []
    for file in os.listdir(base_dir):
        if file.endswith(".txt"):
            data = open(os.path.join(base_dir, file), encoding="ansi").read()
            data = clean_string(data)
            data = data.lower().split('\n')
            song = ''
            if verses > 0:
                count = verses
            else:
                verses = -1
                count = -1
            verse = ''
            for seq in data:
                if '[' in seq or ']' in seq or not seq:
                    continue  # Ignore the chorus and verse tags
                else:
                    # Strip punctuation
                    seq = seq.translate(str.maketrans('', '', string.punctuation))
                    if with_new_line:
                        seq += ' \n'
                    song += seq
                    if count == 0:
                        verse_array.append(verse)
                        count = verses
                        verse = ''
                    if count > 0:
                        verse += seq
                        count -= 1
                    corpus.append(seq)
            if verse != '':
                verse_array.append(verse)
                count = verses
            songs.append(song)
    if verses > 0:
        return verse_array
    return corpus


def create_testing_model(total_words, max_sequence_len):
    model = Sequential()
    model.add(Embedding(total_words, 400, input_length=max_sequence_len - 1))
    model.add(LSTM(300))
    model.add(Dense(total_words, activation='softmax'))
    adam = Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())
    return model


# Create tokenizer and fitted on to the corups passed in
def create_tokenizer(corpus, with_new_line):
    # Tokenize
    if with_new_line:
        tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t')
    else:
        tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(corpus)
    return tokenizer


# Saves anything to a pickle file. Originally just used for tokenizers, but then I pickled input sequences too
def save_tokenizer(name, tokenizer):
    # save tokenizer
    with open(os.path.join(model_dir, name), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Load a pickle file
def load_tokenizer(name):
    # load tokenizer
    with open(os.path.join(model_dir, name), 'rb') as handle:
        tokenizer = pickle.load(handle)
    return tokenizer


def generated_samples_with_model(model_name, tokenizer_name, input_seq, starting_text='Started from the',
                                 temperatures=[0.1, 0.5, 1], word_count=100):
    global songs
    tokenizer = load_tokenizer(tokenizer_name)
    input_sequences = load_tokenizer(input_seq)
    max_sequence_len = max([len(x) for x in input_sequences])

    model = load_model(os.path.join(model_dir, model_name))
    starting_text = clean_string(starting_text)
    starting_text = starting_text.lower()
    starting_text = starting_text.translate(str.maketrans('', '', string.punctuation))
    for temperature in temperatures:
        print('\nTemperature:', temperature)
        generate_stochastic_sampling(starting_text, word_count, temperature, model,
                                     tokenizer,
                                     max_sequence_len)


def train(model_name, tokenizer_name, seq_name):
    global songs
    corpus = read_text_from_corpus(True, verses=5)
    # # Limit corpus to 5k
    # corpus = [:5000]
    # print(corpus[5])
    # tokenizer = create_tokenizer(corpus, True)
    # save_tokenizer(tokenizer_verse_1, tokenizer)
    tokenizer = load_tokenizer(tokenizer_name)
    total_words = len(tokenizer.word_index) + 1
    # # Tokenize sequences and produce n grams
    # input_sequences = []
    # # Line by Line or the verses by verses
    # for line in corpus:
    #     token_list = tokenizer.texts_to_sequences([line])[0]
    #     for i in range(1, len(token_list)):
    #         n_gram_sequence = token_list[:i + 1]
    #         input_sequences.append(n_gram_sequence)
    # # Instead whole body of each song maybe this will give us rhyming
    # for song in songs:
    #     token_list = tokenizer.texts_to_sequences([song])[0]
    #     for i in range(1, len(token_list)):
    #         n_gram_sequence = token_list[:i + 1]
    #         input_sequences.append(n_gram_sequence)
    # # Pickle input Sequences
    # save_tokenizer(input_sequence_verses_5, input_sequences)
    # Load input sequences pickles
    input_sequences = load_tokenizer(seq_name)
    # pad sequences
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
    # create predictors and label
    xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
    print(len(xs))
    print(total_words)
    print(max_sequence_len)
    ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

    # Create Callback to save after every epoch
    callbacks = [
        keras.callbacks.ModelCheckpoint(os.path.join(model_dir, model_name))
    ]
    # Create model
    model = create_testing_model(total_words, max_sequence_len)
    # Train Model
    history = model.fit(xs, ys, epochs=100, batch_size=64, callbacks=callbacks)  # verbose = 1
    model.save(os.path.join(model_dir, model_name))
    print('model saved')

    # Load and retrain
    # model = load_model(os.path.join(model_dir, model_c))
    # history = model.fit(xs, ys, epochs=1, batch_size=128, verbose=1)
    # model.save(os.path.join(model_dir, 'single_bi_lstm1.h5'))
    # print('model saved')


if __name__ == "__main__":
    # train(model_h_full, tokenizer_full, input_sequence_full)
    # read_and_save_corpus()
    temp = [.1, .25, .5, .75, 1, 2, 5, 10]
    generated_samples_with_model(model_g_full, tokenizer_full, input_sequence_full,
                                 'And now I know when that hotline blings\nthat could', temp, 100)
    # tokenizer = load_tokenizer(tokenizer_verse_1)
    # for word, index in tokenizer.word_index.items():
    #     if word is ' ':
    #         print("HAVE SPACE WORD")
    #     elif word is '\n':
    #         print("Have new Line word")
    #     elif ' ' in word:
    #         print("space in word", word)
    #     elif '\n' in word:
    #         print("new line in word", word)
