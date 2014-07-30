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

def write_tree(words, hierarchy, filename):
  f = open(filename, "w")
  print >> f, len(words)
  for word in words:
    print >> f, word

  print >> f, len(hierarchy)
  for children in hierarchy:
    print >> f, len(children), " ".join(map(str, children))

def convert(cluster_data):
  hierarchy = []
  for n1, n2, _, _ in cluster_data:
    hierarchy.append((int(n1), int(n2)))
  return hierarchy

def depth(words, hierarchy):
  depths = [1] * len(words)
  for children in hierarchy:
    new_depth = 1
    for child in children:
      new_depth = max(new_depth, depths[child] + 1)
    depths.append(new_depth)
  return depths[-1]

def read_tree(filename):
  f = open(filename, "r")
  words = []
  num_words = int(f.readline().strip())
  for i in range(num_words):
    words.append(f.readline().strip())

  tree = []
  num_merges = int(f.readline().strip())
  for i in range(num_merges):
    children = map(int, f.readline().strip().split())
    tree.append(tuple(children[1:]))

  return words, tree
