#!/usr/bin/python
from __future__ import division
import sys, getopt
from collections import defaultdict
import re

"""
Merges the trained CLDC outputs back to the document level:
  1) Input file:  L2D10S3 1:0.3 2:0.1 3:0.1231 
  2) Output file: 2 1:0.24 2:0.3 ...
  Essentially, merge the sentence level vectors.
"""
def usage():
  print 'cldc_merger.py input output'

try:
  opts, args = getopt.getopt(sys.argv[1:], "h:", ["help"])
except getopt.GetoptError:
  usage()
  sys.exit(2)
for opt, arg in opts:
  if opt in ("-h", "--help"):
    usage()
    sys.exit()

infile=open(args[0],'r')  
offile=open(args[1],'w')

values = defaultdict(float)

last_doc = 1
last_sen = 0
last_lbl = 0

for line in infile:
  tokens = line.split()
  labelstr = tokens[0] # LiiDjjSkk
  m = re.match(r"L(\d+)D(\d+)S(\d+)", labelstr)
  label = int(m.group(1))
  doc = int(m.group(2))
  sent = int(m.group(3))
  
  if (doc > last_doc):
    offile.write("%s"%last_lbl)
    for i in range(1,len(tokens)):
      values[i] = values[i] / last_sen
      offile.write(" %d:%f"%(i,values[i]))
    offile.write("\n")
    values.clear()

  for i in range(1,len(tokens)):
    val = tokens[i].split(":")[1]
    values[i] += float(val)

  last_lbl = label
  last_doc = doc
  last_sen = sent

offile.write("%s"%last_lbl)
for i in range(1,len(tokens)):
  values[i] = values[i] / last_sen
  offile.write(" %d:%f"%(i,values[i]))
offile.write("\n")
values.clear()
