"""
Clusters word vectors and outputs a tree hierarchy.
"""

# See: http://danifold.net/fastcluster.html?section=3
import fastcluster

from optparse import OptionParser
from utils import *

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

<<<<<<< HEAD
  cluster_data = fastcluster.linkage(
      vectors, method=options.method, metric=options.metric)
  hierarchy = convert_fastcluster(cluster_data)
=======
  hierarchy = fastcluster.linkage(
      vectors, method=options.method, metric=options.metric)
>>>>>>> Script for creating tree files.
  print "Tree depth:", depth(words, hierarchy)

  write_tree(words, hierarchy, options.output_file)


if __name__ == "__main__":
  main()
