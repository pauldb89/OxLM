"""
Print words in each cluster.
"""

from optparse import OptionParser
from utils import *

def main():
  parser = OptionParser()
  parser.add_option("-i", "--input", dest="input_file", help="Input file")
  options, _ = parser.parse_args()

  words, tree = read_tree(options.input_file)
  words = [[word] for word in words]
  for children in tree:
    new_cluster = []
    for child in children:
      new_cluster.extend(words[child])
    words.append(new_cluster)
    print new_cluster
    raw_input()


if __name__ == "__main__":
  main()
