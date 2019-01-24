from keras.preprocessing.text import Tokenizer
import numpy as np
import collections
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing import sequence
import json
from tqdm import tqdm
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Bidirectional, Conv1D, MaxPooling1D, Dropout, Dense, concatenate, Input, Reshape, Flatten
from keras.models import Model
from keras.callbacks import EarlyStopping


def load_embeddings():
    # load pre-trained word embeddings (GloVe, general crawl)
    embeddings_index = {}
    with open('glove.840B.300d.txt') as f:
        for line in f:
            values = line.split()
            word = values[0]

            try:
                coefs = np.asarray(values[1:], dtype='float32')
            except:
                pass

            embeddings_index[word] = coefs
    return embeddings_index


Sentence = collections.namedtuple("Sentence", ["id", "sentence", "prev_1", "prev_2", "prev_3"])


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


# load validation and training data
train_X, train_Y = load_data("da_tagging/utterances.train", True)
valid_X, valid_Y = load_data("da_tagging/utterances.valid", True)

# represent labels as vectors
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

# create an embedding matrix for the words we have in the dataset
embeddings_index = load_embeddings()  # load GloVe embeddings
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

target_input = Input(shape=(max_len, ))

# target sentence - LSTM - CNN - Max pooling
# turns indices into embedding vectors
e = Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], input_length=max_len, trainable=False)(target_input)
b = Bidirectional(LSTM(300, dropout=0.5, recurrent_dropout=0.5))(e)
br = Reshape((600, 1))(b)
c1 = Conv1D(6, kernel_size=3, activation='relu', input_shape=(None, 600, 1))(br) # change params
m = MaxPooling1D(pool_size=2)(c1)
m = Flatten()(m)

# contexts
X_train_1_seq = token.texts_to_sequences(X_train_1)
X_valid_1_seq = token.texts_to_sequences(X_valid_1)

X_train_1_pad = sequence.pad_sequences(X_train_1_seq, maxlen=82)
X_valid_1_pad = sequence.pad_sequences(X_valid_1_seq, maxlen=82)

input_1 = Input(shape=(max_len,))

e1 = Embedding(len(word_index) + 1, 300, weights=[embedding_matrix], input_length=max_len, trainable=False)(input_1)
b1 = Bidirectional(LSTM(300, dropout=0.5, recurrent_dropout=0.5))(e1)

# b1 = Dense(1024, activation='relu')(b1)
# b1 = Dropout(0.8)(b1)

# the further -> the higher dropout? or some other things? how to represent that the furthert, the less important??

concatenated = concatenate([m, b1], axis=1)
out = Dense(31, activation='sigmoid')(concatenated) # try softmax too

model = Model([target_input, input_1], out)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model with early stopping callback
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

model.fit([X_train_pad, X_train_1_pad], y=train_Y_bi, batch_size=512, epochs=500,
          verbose=1, validation_data=([X_valid_pad, X_valid_1_pad], valid_Y_bi), callbacks=[earlystop])

# test_X, _ = load_data("da_tagging/utterances.test", False)
