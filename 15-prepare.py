import glob

import json

import random

import copy
char_index_0 = json.loads(open('char_index_000000000.json').read())
char_index_1 = json.loads(open('char_index_000000001.json').read())

chars = list('abcdefghijklmnopqrstuvwxyz ,.')

total = ''
for name in glob.glob('courpus/bbc/*/*'):
  try:
    text = open(name).read()
  except Exception as ex:
    continue
  buff = []
  for char in list(text.lower()):
    if char in char_index_0:
      buff.append(char)
  total += ''.join(buff)

# random slice 
pairs = []
for index in random.sample( list(range(0, len(total) - 150)),100000):
  _char_index_0 = copy.copy(char_index_0)
  _char_index_1 = copy.copy(char_index_1)
  real = total[index:index+150]

  enigma = []
  for diff, char in enumerate(real):
    # roater No.1 update _char_index
    _char_index_0 = { char:(ind+1)%len(_char_index_0) for char, ind in _char_index_0.items() }
    # get index
    ind = _char_index_0[char]
    next_char = chars[ind]

    # roater No.2 
    _char_index_1 = { char:(ind+1)%len(_char_index_1) for char, ind in _char_index_1.items() }
    # get index
    ind = _char_index_1[next_char]
    next_char = chars[ind]
    
    enigma.append(next_char)
  cript = ''.join(enigma)

  crop = random.choice(list(range(len(char_index_0))))
  real, cript = real[crop:crop+100], cript[crop:crop+100]
  pairs.append( (real, cript) )

open('pairs.json', 'w').write( json.dumps(pairs, indent=2) )
