from keras.layers          import Lambda, Input, Dense, GRU, LSTM, RepeatVector
from keras.models          import Model
from keras.layers.core     import Flatten
from keras.callbacks       import LambdaCallback 
from keras.optimizers      import SGD, RMSprop, Adam
from keras.layers.wrappers import Bidirectional as Bi
from keras.layers.wrappers import TimeDistributed as TD
from keras.layers          import merge, multiply
from keras.regularizers    import l2
from keras.layers.core     import Reshape, Dropout
from keras.layers.normalization import BatchNormalization as BN
import keras.backend as K
import numpy as np
import random
import sys
import pickle
import glob
import copy
import os
import re

timesteps   = 100
inputs      = Input(shape=(timesteps, 29))
x           = Bi(GRU(512, dropout=0.10, recurrent_dropout=0.25, return_sequences=True))(inputs)
x           = TD(Dense(3000, activation='relu'))(x)
x           = Dropout(0.2)(x)
x           = TD(Dense(3000, activation='relu'))(x)
x           = Dropout(0.2)(x)
x           = TD(Dense(29, activation='softmax'))(x)

decript     = Model(inputs, x)
decript.compile(optimizer=Adam(), loss='categorical_crossentropy')

buff = None
def callbacks(epoch, logs):
  global buff
  buff = copy.copy(logs)
  print("epoch" ,epoch)
  print("logs", logs)

def train():
  vectors = pickle.loads(open('vectors.pkl', 'rb').read())

  Xs, ys = [], []
  for v in vectors[:99000]:
    X,y = v
    Xs.append(X)
    ys.append(y)
  Xs = np.array( Xs )
  ys = np.array( ys )
  print(Xs.shape)
  if '--resume' in sys.argv:
    model = sorted( glob.glob("models/*.h5") ).pop(0)
    print("loaded model is ", model)
    decript.load_weights(model)

  for i in range(5):
    print_callback = LambdaCallback(on_epoch_end=callbacks)
    batch_size = random.randint( 32, 64 )
    random_optim = random.choice( [Adam(), RMSprop()] )
    print( random_optim )
    decript.optimizer = random_optim
    decript.fit( Xs, ys,  shuffle=True, batch_size=batch_size, epochs=1, callbacks=[print_callback] )
    decript.save("models/%9f_%09d.h5"%(buff['loss'], i))
    print("saved ..")
    print("logs...", buff )

def predict():
  vectors = pickle.loads(open('vectors.pkl', 'rb').read())
  Xs, ys = [], []
  for v in vectors[99000:]:
    X,y = v
    Xs.append(X)
    ys.append(y)
  Xs = np.array( Xs )
  ys = np.array( ys )
  
  decript.load_weights( sorted(glob.glob('models/*')).pop(0) )
  yps = decript.predict(Xs)
  chars = list('abcdefghijklmnopqrstuvwxyz ,.')
  index_char = {index:char for index, char in enumerate(chars)}
  
  for yp, yo, xs in zip(yps.tolist(), ys.tolist(), Xs.tolist()):
    predict = ''.join([index_char[np.argmax(y)] for y in yp] )
    inputs = ''.join([index_char[np.argmax(x)] for x in xs] )
    origin = ''.join([index_char[np.argmax(y)] for y in yo] )
    print('origin', origin)
    print('inputs', inputs)
    print('predict', predict)
    print()
  
if __name__ == '__main__':

  if '--train' in sys.argv:
    train()

  if '--predict' in sys.argv:
    predict()
