"""
Sample lines from monolingual corpora to create smaller back-off n-gram models.
"""

import sys

from optparse import OptionParser
from random import random

def main():
  parser = OptionParser()
  parser.add_option("-i", "--input", dest="input", help="Input file")
  parser.add_option("-o", "--output", dest="output", help="Output file")
  parser.add_option("-r", "--rate", dest="rate", help="Sampling rate")
  options, _ = parser.parse_args()

  f = open(options.output, "w")
  rate = float(options.rate)
  line_index = 0
  for line in open(options.input):
    if random() <= rate:
      print >> f, line.strip()

    line_index += 1
    if line_index % 100000 == 0:
      sys.stderr.write(".")
      if line_index % 10000000 == 0:
        print "[", line_index, "]"


if __name__ == "__main__":
  main()
