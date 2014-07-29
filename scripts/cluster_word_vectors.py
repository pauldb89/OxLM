"""
Clusters word vectors and outputs a tree hierarchy.
"""

# See: http://danifold.net/fastcluster.html?section=3
import fastcluster

from optparse import OptionParser
from utils import *

def read_words(filename):
  words = []
  for line in open(filename):
    words.append(line.strip())
  return words

def read_vectors(filename):
  vectors = []
  for line in open(filename):
    vectors.append(map(float, line.strip().split()))
  return vectors

def depth(words, hierarchy):
  depths = [1] * len(words)
  for node1, node2, _, _ in hierarchy:
    depths.append(1 + max(depths[int(node1)], depths[int(node2)]))
  return depths[-1]

def write_tree(words, hierarchy, filename):
  f = open(filename, "w")
  print >> f, len(words)
  for word in words:
    print >> f, word

  print >> f, len(hierarchy)
  for node1, node2, _, _ in hierarchy:
    print >> f, 2, int(node1), int(node2)

def main():
  parser = OptionParser()
  parser.add_option("-d", "--vocab-file", dest="vocab_file",
                    help="File containing the vocabulary")
  parser.add_option("-v", "--vectors-file", dest="vectors_file",
                    help="File containing the word_vectors")
  parser.add_option("-m", "--method", dest="method", help="Clustering method")
  parser.add_option("-s", "--metric", dest="metric", help="Clusterig metric")
  parser.add_option("-o", "--output-file", dest="output_file",
                    help="File containing the word hierarchy")
  options, _ = parser.parse_args()

  words = read_words(options.vocab_file)
  print "Read", len(words), "words..."

  vectors = read_vectors(options.vectors_file)
  print "Read", len(vectors), "vectors..."

  cluster_data = fastcluster.linkage(
      vectors, method=options.method, metric=options.metric)
  hierarchy = convert_fastcluster(cluster_data)
  print "Tree depth:", depth(words, hierarchy)

  write_tree(words, hierarchy, options.output_file)


if __name__ == "__main__":
  main()
