"""
Bottom-up algorithm that greedly flattens a tree up to some branching factor.
"""

from optparse import OptionParser
from utils import *

def compute_min_depth(words, tree, branch_factor):
  dp = []
  min_depth = []
  source = []
  num_nodes = len(words) + len(tree)
  b = branch_factor + 1
  for i in range(len(words)):
    dp.append([[num_nodes + 1] * (branch_factor + 1)])
    dp[i][0][1] = 1
    source.append([[0] * (branch_factor + 1)])
    min_depth.append(dp[i][0])

  for x in range(len(tree)):
    i = len(words) + x

    dp.append([[num_nodes + 1] * (branch_factor + 1) for j in range(len(tree[x]) + 1)])
    source.append([[0] * (branch_factor + 1) for j in range(len(tree[x]) + 1)])
    dp[i][0][0] = 0

    for j, child in enumerate(tree[x]):
      for k in range(branch_factor):
        for alpha in range(1, branch_factor + 1 - k):
          val = max(dp[i][j][k], min_depth[child][alpha])
          if val < dp[i][j + 1][k + alpha]:
            dp[i][j + 1][k + alpha] = val
            source[i][j + 1][k + alpha] = (alpha, 0)

        for alpha in range(1, branch_factor + 1):
          val = max(dp[i][j][k], 1 + min_depth[child][alpha])
          if val < dp[i][j + 1][k + 1]:
            dp[i][j + 1][k + 1] = val
            source[i][j + 1][k + 1] = (alpha, 1)

    min_depth.append(dp[i][len(tree[x])])

  return min_depth, source

def construct_tree(words, tree, source, node, size, new_tree):
  if node < len(words):
    return [node]

  new_children = []
  node_index = node - len(words)
  for i in reversed(range(len(tree[node_index]))):
    child = tree[node_index][i]
    alpha, op = source[node][i + 1][size]
    if op == 0:
      children = construct_tree(words, tree, source, child, alpha, new_tree)
      new_children.extend(children)
      size -= alpha
    else:
      children = construct_tree(words, tree, source, child, alpha, new_tree)
      new_children.append(len(words) + len(new_tree))
      new_tree.append(children)
      size -= 1

  return new_children

def get_tree(words, tree, branch_factor, min_depth, source):
  num_nodes = len(words) + len(tree)
  size = 0
  tree_depth = num_nodes + 1
  for i in range(branch_factor + 1):
    if min_depth[num_nodes - 1][i] + 1 < tree_depth:
      tree_depth = min_depth[num_nodes - 1][i] + 1
      size = i
  print "Tree depth:", tree_depth

  new_tree = []
  new_children = construct_tree(words, tree, source, num_nodes - 1, size, new_tree)
  new_tree.append(new_children)

  assert depth(words, new_tree) == tree_depth

  return new_tree

def flatten_tree(words, tree, branch_factor):
  for children in tree:
    if len(children) > branch_factor:
      raise Error("The tree is already too flat")

  min_depth, source = compute_min_depth(words, tree, branch_factor)
  new_tree = get_tree(words, tree, branch_factor, min_depth, source)

  return new_tree

def main():
  parser = OptionParser()
  parser.add_option("-i", "--input", dest="input_file",
                    help="File containing input tree")
  parser.add_option("-b", "--branch-factor", dest="branch_factor",
                    help="Branching factor")
  parser.add_option("-o", "--output", dest="output_file",
                    help="File containing output tree")
  options, _ = parser.parse_args()

  words, tree = read_tree(options.input_file)
  branch_factor = int(options.branch_factor)
  new_tree = flatten_tree(words, tree, branch_factor)
  write_tree(words, new_tree, options.output_file)


if __name__ == "__main__":
  main()
