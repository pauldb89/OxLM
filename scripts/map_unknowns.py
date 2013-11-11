#!/usr/bin/python
import sys, getopt
from collections import defaultdict

def usage():
  print 'map_unknowns.py [--cutoff n] file [test_file]'

cutoff = 2
try:                                
  opts, args = getopt.getopt(sys.argv[1:], "hc:", ["help","cutoff="])
except getopt.GetoptError:          
  usage()                         
  sys.exit(2)                     
for opt, arg in opts:                
  if opt in ("-h", "--help"):      
    usage()                         
    sys.exit()                  
  elif opt in ('-c', "--cutoff"):                
    cutoff = int(arg)

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
    if word_freq[token] < cutoff: print "_UNK_",
    else:                         print token,
  print
