from keras.preprocessing.text import Tokenizer
import numpy as np
import collections
from sklearn.preprocessing import LabelBinarizer
import json
from tqdm import tqdm
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dropout, Dense, concatenate, Input, Reshape,\
    Flatten, BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
from keras.models import model_from_json
from keras import regularizers
from keras.preprocessing import text as txt, sequence

Sentence = collections.namedtuple("Sentence", ["id", "sentence", "prev_1", "prev_2", "prev_3"])


def load_embeddings():
    embeddings_index = {}
    with open('glove.twitter.27B/glove.twitter.27B.100d.txt') as f:
        for line in f:
            values = line.split()
            word = values[0]

            try:
                coefs = np.asarray(values[1:], dtype='float32')
            except:
                pass

            embeddings_index[word] = coefs
    return embeddings_index


def load_data(file_name, labeled):
    X = []
    Y = []
    with open(file_name) as f:
        for l in f:
            if labeled:
                id, label, sentes = l.split("\t")
                Y.append(label)
            else:
                id, sentes = l.split('\t')
            prev_3, prev_2, prev_1, sent = sentes.split(";")
            X.append(Sentence(id=id, sentence=sent, prev_1=prev_1, prev_2=prev_2, prev_3=prev_3))

    return X, Y


def convert_text_to_index_array(text, dictionary):
    words = txt.text_to_word_sequence(text)
    word_indices = []
    for word in words:
        if word in dictionary:
            word_indices.append(dictionary[word])
        else:
            print("'%s' not in training corpus; ignoring." % word)
    return word_indices


# imitates keras tokenizer used during training
def normalize_text(text):
    text = text.lower()
    filters = '"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n'
    split = " "
    for c in filters:
        text = text.replace(c, split)
    return text


def train():
    train_X, train_Y = load_data("da_tagging/utterances.train", True)
    valid_X, valid_Y = load_data("da_tagging/utterances.valid", True)

    lb = LabelBinarizer()
    lb.fit(train_Y)
    train_Y_bi = lb.fit_transform(train_Y)
    valid_Y_bi = lb.fit_transform(valid_Y)

    X_train_ids, X_train, X_train_1, X_train_2, X_train_3 = zip(*train_X)
    X_valid_ids, X_valid, X_valid_1, X_valid_2, X_valid_3 = zip(*valid_X)

    token = Tokenizer(filters='"#$%&()*+,.-/:;<=>@[\\]^_`{|}~\t\n')  # keep ! and ?
    token.fit_on_texts(
        list(X_train) + list(X_train_1) + list(X_train_2) + list(X_train_3) + list(X_valid) + list(X_valid_1) +
        list(X_valid_2) + list(X_valid_3))

    X_train_seq = token.texts_to_sequences(X_train)
    X_valid_seq = token.texts_to_sequences(X_valid)

    max_len = 82
    X_train_pad = sequence.pad_sequences(X_train_seq, maxlen=82)
    X_valid_pad = sequence.pad_sequences(X_valid_seq, maxlen=82)

    word_index = token.word_index
    # save dictionary to use it for prediction
    with open('dictionary.json', 'w') as dictionary_file:
        json.dump(word_index, dictionary_file)

    embeddings_index = load_embeddings()
    embedding_matrix = np.zeros((len(word_index) + 1, 100))
    count = 0
    for word, i in tqdm(word_index.items()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.uniform(-0.25, 0.25, 100)
            count += 1
    print("words not in embeddings:" + str(count))

    target_input = Input(shape=(max_len,))

    e = Embedding(len(word_index) + 1, 100, weights=[embedding_matrix], input_length=max_len, trainable=True)

    e1 = e(target_input)

    # contexts
    X_train_1_pad, X_valid_1_pad, b1, input_1 = context_layer(X_train_1, X_valid_1, max_len, token, 1, e)
    X_train_2_pad, X_valid_2_pad, b2, input_2 = context_layer(X_train_2, X_valid_2, max_len, token, 2, e)
    # X_train_3_pad, X_valid_3_pad, b3, input_3 = context_layer(X_train_3, X_valid_3, max_len, token, 3, e)

    b = Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.5))(e1)
    br = Reshape((256, 1))(b)

    filter_lengths = [2, 3, 4]
    for i in filter_lengths:
        br = Conv1D(50, kernel_size=i)(br)

        br = MaxPooling1D(pool_size=4)(br)
    br = GlobalMaxPooling1D()(br)
    br = BatchNormalization()(br)

    # br = Flatten()(br)

    concatenated = concatenate([br, b1, b2], axis=1)

    out = Dense(31, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(concatenated)

    model = Model([target_input, input_1, input_2], out)
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=['accuracy'])

    earlystop = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=0, mode='auto', restore_best_weights=True)
    checkpoint = ModelCheckpoint("weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5", monitor='val_acc', verbose=1,
                                 save_best_only=True, mode='max')

    model.fit([X_train_pad, X_train_1_pad, X_train_2_pad], y=train_Y_bi, batch_size=128, epochs=50,
              verbose=1, validation_data=([X_valid_pad, X_valid_1_pad, X_valid_2_pad], valid_Y_bi),
              callbacks=[earlystop, checkpoint])

    model_json = model.to_json()
    with open('model.json', 'w') as json_file:
        json_file.write(model_json)

    model.save_weights('model.h5')


def context_layer(X_train_1, X_valid_1, max_len, token, n, e):
    X_train_1_seq = token.texts_to_sequences(X_train_1)
    X_valid_1_seq = token.texts_to_sequences(X_valid_1)
    X_train_1_pad = sequence.pad_sequences(X_train_1_seq, maxlen=82)
    X_valid_1_pad = sequence.pad_sequences(X_valid_1_seq, maxlen=82)

    input_1 = Input(shape=(max_len,))
    e1 = e(input_1)
    b1 = Bidirectional(LSTM(128, dropout=0.5, recurrent_dropout=0.5))(e1)

    b1 = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(b1)
    if n == 1:
        b1 = Dropout(0.6)(b1)
    elif n == 2:
        b1 = Dropout(0.7)(b1)
    elif n == 3:
        b1 = Dropout(0.8)(b1)

    return X_train_1_pad, X_valid_1_pad, b1, input_1


def predict():
    test_X, _ = load_data("da_tagging/utterances.test", False)

    train_X, train_Y = load_data("da_tagging/utterances.train", True)
    valid_X, valid_Y = load_data("da_tagging/utterances.valid", True)

    labels = list(set(train_Y + valid_Y))

    # load saved model
    with open('model.json', 'r') as f:
        loaded_model_json = f.read()

    model = model_from_json(loaded_model_json)
    model.load_weights('weights-improvement-15-0.65.hdf5')

    # load dictionary saved during training to use it for converting text to number vectors
    with open('dictionary.json', 'r') as dictionary_file:
        dictionary = json.load(dictionary_file)

    with open('result.txt', 'w') as r:
        for x in test_X:
            input = []
            i = 0
            id = ""
            for X in x:
                if i == 0:
                    id = X
                    i += 1
                    continue

                if i > 3:
                    break

                text = normalize_text(X)
                words = convert_text_to_index_array(text, dictionary=dictionary)
                words_pad = sequence.pad_sequences([words], maxlen=82)
                input.append(words_pad)

                i += 1

            pred = model.predict(input)

            y_class = pred.argmax(axis=-1)
            predicted_label = sorted(labels)[y_class.item()]
            r.write(id + '\t' + predicted_label + '\n')


if __name__ == '__main__':
    model = Path("model.h5")
    if model.exists():
        predict()
    else:
        train()
