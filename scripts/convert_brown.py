"""
Convert tree from the format used by the Brown clustering tool.
https://github.com/percyliang/brown-cluster
"""

from optparse import OptionParser
from utils import *

def read_brown_clusters(filename):
  data = {}
  for line in open(filename):
    items = line.strip().split()
    data[items[0]] = items[1]

  return data

def construct_tree(data, label, words, tree):
  if label in data:
    words.append(data[label])
    return len(words) - 1

  left_child = construct_tree(data, label + "0", words, tree)
  right_child = construct_tree(data, label + "1", words, tree)

  tree.append((left_child, right_child))

  return len(data) + len(tree) - 1

def convert_brown_clusters(data):
  tree = []
  words = []
  construct_tree(data, "", words, tree)
  return words, tree

def main():
  parser = OptionParser()
  parser.add_option("-i", "--input", dest="input_file", help="Input file")
  parser.add_option("-o", "--output", dest="output_file", help="Output file")
  options, _ = parser.parse_args()

  data = read_brown_clusters(options.input_file)
  words, tree = convert_brown_clusters(data)
  print "Tree depth:", depth(words, tree)
  write_tree(words, tree, options.output_file)

if __name__ == "__main__":
  main()
