"""

IMDB - Sentiment Analysis
This project uses an LSTM (rnn) on the IMDB dataset to predict sentiment classification

Train on 25000 samples, validate on 25000 samples
Epoch 1/12
2020-02-24 18:24:19.529793: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
25000/25000 [==============================] - 279s 11ms/step - loss: 0.4600 - acc: 0.7845 - val_loss: 0.4473 - val_acc: 0.7944
Epoch 2/12
25000/25000 [==============================] - 242s 10ms/step - loss: 0.2953 - acc: 0.8812 - val_loss: 0.3590 - val_acc: 0.8424
Epoch 3/12
25000/25000 [==============================] - 261s 10ms/step - loss: 0.2156 - acc: 0.9159 - val_loss: 0.4556 - val_acc: 0.8289
Epoch 4/12
25000/25000 [==============================] - 297s 12ms/step - loss: 0.1534 - acc: 0.9428 - val_loss: 0.4212 - val_acc: 0.8392
Epoch 5/12
25000/25000 [==============================] - 258s 10ms/step - loss: 0.1086 - acc: 0.9610 - val_loss: 0.5586 - val_acc: 0.8292
Epoch 6/12
25000/25000 [==============================] - 270s 11ms/step - loss: 0.0795 - acc: 0.9727 - val_loss: 0.5739 - val_acc: 0.8262
Epoch 7/12
25000/25000 [==============================] - 301s 12ms/step - loss: 0.0588 - acc: 0.9802 - val_loss: 0.6307 - val_acc: 0.8261
Epoch 8/12
25000/25000 [==============================] - 269s 11ms/step - loss: 0.0512 - acc: 0.9826 - val_loss: 0.7165 - val_acc: 0.8306
Epoch 9/12
25000/25000 [==============================] - 262s 10ms/step - loss: 0.0344 - acc: 0.9889 - val_loss: 0.7766 - val_acc: 0.8269
Epoch 10/12
25000/25000 [==============================] - 251s 10ms/step - loss: 0.0314 - acc: 0.9897 - val_loss: 0.8213 - val_acc: 0.8254
Epoch 11/12
25000/25000 [==============================] - 251s 10ms/step - loss: 0.0235 - acc: 0.9925 - val_loss: 0.8301 - val_acc: 0.8289
Epoch 12/12
25000/25000 [==============================] - 253s 10ms/step - loss: 0.0151 - acc: 0.9956 - val_loss: 0.8599 - val_acc: 0.8205
25000/25000 [==============================] - 75s 3ms/step
Model accuracy: 0.82048
text: This movie was an amazing experience. Loved it!
sent score: 0.9999765157699585
classification: positive

text: Best movie I've seen in a while.
sent score: 0.9993841648101807
classification: positive


text: I fell asleep.
sent score: 0.1708667129278183
classification: negative

"""

import re
import json


import numpy as np
import keras
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding,Dense,Dropout,LSTM
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer


def main():
    """ build and train our model"""
    
    max_features = 20000
    max_len = 100
    batch_size = 32
    tokenizer = Tokenizer(num_words=max_features, split=' ')


    (x_train, y_train), (x_test, y_test) = define_dataset(max_features=max_features, max_len=max_len)
    model = define_model(max_features)
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=12,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size)
    print('Model accuracy:', acc)

    #initialize imdb indices from same datasource
    path = keras.utils.get_file(
        "imdb_word_index.json",
        origin='https://s3.amazonaws.com/text-datasets/imdb_word_index.json',
        file_hash='bfafd718b763782e994055a2d397834f')
    with open(path) as f:
        word_indices = json.load(f)
    
    #test on a few handmade sequences
    test_sequences = [
            "This movie was an amazing experience. Loved it!",
            "Best movie I've seen in a while.",
            "I fell asleep.",
    ]
    
    #precompile regex to save computation on each test
    pattern_re = re.compile("([\w][\w']*\w)")
    remove_re = re.compile('[^a-zA-Z]')
    
    for seq in test_sequences:
        vec = sequence2vec(seq, word_indices, max_features=max_features, pattern=pattern_re, remove=remove_re) 
        vec = pad_sequences([vec], maxlen=max_len, dtype='int32', value=0)
        sentiment = model.predict(vec,batch_size=1,verbose = 2)[0]
        cls = None
        sent_score = sentiment[0]
        if(round(sent_score) == 0):
            cls = "negative"
        elif (round(sent_score) == 1):
            cls = "positive"
        print("text: {} \nsent score: {} \nclassification: {} \n\n ".format(seq,sentiment[0],cls)) 

def define_dataset(max_len=80, max_features=20000):

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

    #pad sequences
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)
    
    #x_train = x_train[0:250]
    #y_train = y_train[0:250]
    #x_test = x_test[0:250]
    #y_test = y_test[0:250]

    return (x_train, y_train), (x_test, y_test) 

def define_model(max_features):
    """ define our LSTM model"""

    model = keras.models.Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model

def sequence2vec(s, indices, 
        max_features=20000,
        pattern=re.compile("([\w][\w']*\w)"),
        remove=re.compile('[^a-zA-Z]')):
    """ using the word index as used in the provided dictionary, turn a string into a vector of word indices"""  
    words = pattern.findall(s) #split into list of words
    vec = []
    for word in words:
        word = remove.sub('',word) #remove any non-alphabet characters from string
        if word in indices:
            idx = indices[word]
            if idx <= max_features:
                vec.append(indices[word])
            else:
                vec.append(indices["unknown"])
        else:
            vec.append(indices["unknown"])
    return np.asarray(vec)

if __name__ == "__main__":
    main()


