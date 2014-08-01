"""
Script which produces the Huffman tree for a given corpus.
"""

from optparse import OptionParser
from collections import defaultdict
import heapq

from utils import *

def count_words(filename):
  word_counts = defaultdict(int)
  word_counts["<s>"] = 0

  for line in open(filename):
    words = line.strip().split()
    for word in words:
      word_counts[word] += 1

    word_counts["</s>"] += 1

  return word_counts


def get_huffman_tree(word_counts):
  words = word_counts.keys()
  heap = [(word_count, i) for i, word_count in enumerate(word_counts.values())]
  heapq.heapify(heap)

  num_nodes = len(words)
  tree = []
  while len(heap) >= 2:
    c1, n1 = heapq.heappop(heap)
    c2, n2 = heapq.heappop(heap)

    tree.append((n1, n2))
    heapq.heappush(heap, (c1 + c2, num_nodes))
    num_nodes += 1

  return words, tree


def main():
  parser = OptionParser()
  parser.add_option("-i", "--input", dest="input_file",
                    help="Input file containing training corpus")
  parser.add_option("-o", "--output", dest="output_file",
                    help="Output file")
  options, _ = parser.parse_args()

  word_counts = count_words(options.input_file)
  print "Vocabulary size:", len(word_counts)

  words, tree = get_huffman_tree(word_counts)
  print "Tree depth:", depth(words, tree)
  write_tree(words, tree, options.output_file)


if __name__ == "__main__":
  main()
