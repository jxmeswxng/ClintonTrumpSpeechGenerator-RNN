import warnings
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import code
import sys
from create_word2vec import *

# load word2vec dictionaries
preprocessing = Word2Vec()
s_lists_sentences_clinton, s_lists_sentences_trump, w2v_dict_clinton, \
 w2v_dict_trump, c_model, t_model = preprocessing()
c_train_data, c_train_labels = preprocessing.generateInputData(
                               s_lists_sentences_clinton, c_model)
trump_train_data, trump_train_labels = preprocessing.generateInputData(
                                       s_lists_sentences_trump, t_model)

# print ("Total unique vocab in Clinton: " + str(len(w2v_dict_clinton)))
print ("Total unique vocab in Trump: " + str(len(w2v_dict_trump)))

# prepare input data to the two models
seq_length = 3
tn_seq = len(trump_train_data)
tX = numpy.reshape(trump_train_data, (tn_seq, seq_length, 300))
ty = numpy.array(trump_train_labels)

# define the LSTM model
model = Sequential()
model.add(LSTM(1200, input_shape=(tX.shape[1:])))
model.add(Dropout(0.3))
model.add(Dense(ty.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1,
                             save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# uncomment "model.fit..." line to train the model
# comment out if running "application"
# model.fit(tX, ty, nb_epoch=100, batch_size=32, callbacks=callbacks_list)

# comment out everything below this line when training model
# uncomment when running "application"
# load the network weights returned from model
filename = "weights-improvement-99--0.4310-bigger.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# a while loop that continuously prompt for user input
while True:
    userInput = input("Enter three words: ").lower().strip().split(" ")
    flag = True
    if len(userInput) != 3:
        print("Incorrect user input.")
        flag = False
    pattern = []
    for w in userInput:
        try:
            w2v = w2v_dict_trump[w]
            pattern.append(w2v)
        except:
            print("User input not in dictionary. Please try again.")
            flag = False

    # generator with yield keyword
    # output the next three words given input
    def lstm(pattern):
        for i in range(3):
            x = numpy.reshape(pattern, (-1, 3, 300))
            # prediction is a matrix returned from model
            prediction = model.predict(x, verbose=0)
            prediction = numpy.reshape(prediction, (300,))
            # find the most similar word given the prediction matrix
            result = t_model.most_similar(positive=[prediction], topn=1)
            word, prob = result[0]
            # convert returned word back to a vector
            # set this vector as the next vector in pattern
            wordVect = w2v_dict_trump[word]
            pattern.append(wordVect)
            pattern = pattern[1:len(pattern)]
            yield word

    if flag is True:
        print ("Trump says:")
        for word in lstm(pattern):
            print(word)
