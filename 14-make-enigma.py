import os

import random

import json

chars = list('abcdefghijklmnopqrstuvwxyz ,.')

for i in range(2):
  indexs = [index for index, char in enumerate(chars)]
  random.shuffle(indexs)

  char_index = {char:index for char, index in zip(chars, indexs)}

  open('char_index_{:09d}.json'.format(i), 'w').write( json.dumps(char_index, indent=2) )


