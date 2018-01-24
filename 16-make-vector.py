import json
import numpy as np
import pickle

chars = list('abcdefghijklmnopqrstuvwxyz ,.')
char_index = {char:index for index, char in enumerate(chars)}

pairs = json.loads(open('pairs.json').read())

vectors = []
for pair in pairs:
  real, cript = pair

  X = [ [0.0]*len(char_index) for i in range(100)]
  for index, c in enumerate(cript):
    X[index][ char_index[c] ] = 1.0
  
  y = [ [0.0]*len(char_index) for i in range(100)]
  for index, r in enumerate(real):
    y[index][ char_index[r] ] = 1.0

  X,y = np.array(X), np.array(y)
  vectors.append( (X,y) )

open('vectors.pkl', 'wb').write(pickle.dumps(vectors))
