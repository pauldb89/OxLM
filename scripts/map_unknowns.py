#!/usr/bin/python
import sys, getopt
from collections import defaultdict

def usage():
  print 'map_unknowns.py [--cutoff n] [--unknown-token string] file [test_file]'

cutoff = 2
unk = "_UNK_"
try:
  opts, args = getopt.getopt(sys.argv[1:], "hcu:", ["help","cutoff=","unknown-token="])
except getopt.GetoptError:
  usage()
  sys.exit(2)
for opt, arg in opts:
  if opt in ("-h", "--help"):
    usage()
    sys.exit()
  elif opt in ('-c', "--cutoff"):
    cutoff = int(arg)
  elif opt in ('-u', "--unknown-token"):
    unk = arg

file=open(args[0],'r')
word_freq = defaultdict(int)

for line in file:
  tokens = line.split()
  for token in tokens:
    word_freq[token] += 1

file=open(args[-1],'r')
for line in file:
  tokens = line.split()
  for token in tokens:
    if word_freq[token] < cutoff: print unk,
    else:                         print token,
  print
