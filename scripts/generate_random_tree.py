"""
Generates a random binary tree hierarchy.
"""

from optparse import OptionParser
from utils import *

import random

def main():
  parser = OptionParser()
  parser.add_option("-v", "--vocab-file", dest="vocab_file",
                    help="File containing the vocabulary")
  parser.add_option("-o", "--output-file", dest="output_file",
                    help="Where to write the word hierarchy")
  options, _ = parser.parse_args()

  words = read_words(options.vocab_file)
  clusters = [i for i in range(len(words))]

  hierarchy = []
  num_nodes = len(words)
  for i in range(len(words) - 1):
    c1, c2 = tuple(random.sample(clusters, 2))
    hierarchy.append((c1, c2))
    clusters.remove(c1)
    clusters.remove(c2)
    clusters.append(num_nodes)
    num_nodes += 1

  print "Tree depth:", depth(words, hierarchy)

  write_tree(words, hierarchy, options.output_file)

if __name__ == "__main__":
  main()
