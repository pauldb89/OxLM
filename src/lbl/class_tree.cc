#include "lbl/class_tree.h"

#include <queue>

namespace oxlm {

ClassTree::ClassTree() {}

ClassTree::ClassTree(
    const string& filename,
    boost::shared_ptr<Vocabulary>& vocab) {
  assert(filename.size());
  ifstream fin(filename);

  int num_words;
  fin >> num_words;
  for (int i = 0; i < num_words; ++i) {
    string word;
    fin >> word;
    vocab->convert(word);
  }

  int num_merges;
  fin >> num_merges;
  int num_nodes = num_words + num_merges;
  vector<vector<int>> tree(num_nodes);
  for (int i = 0; i < num_merges; ++i) {
    int num_values;
    fin >> num_values;
    for (int j = 0; j < num_values; ++j) {
      int node;
      fin >> node;
      tree[num_words + i].push_back(node);
    }
  }

  // Relabel tree such that children have consecutive labels.
  int num_labels = 0;
  parent.resize(num_nodes, -1);
  children.resize(num_nodes);
  index.resize(num_words);

  queue<pair<int, int>> q;
  q.push(make_pair(num_nodes - 1, 0));
  while (!q.empty()) {
    int node = q.front().first;
    int label = q.front().second;
    q.pop();

    for (int child: tree[node]) {
      ++num_labels;
      children[label].push_back(num_labels);
      parent[num_labels] = label;
      q.push(make_pair(child, num_labels));
    }

    if (tree[node].size() == 0) {
      index[node] = label;
    }
  }
}

size_t ClassTree::size() const {
  return parent.size();
}

int ClassTree::getRoot() const {
  return 0;
}

int ClassTree::getNode(int word_id) const {
  return index[word_id];
}

int ClassTree::getParent(int node) const {
  return parent[node];
}

vector<int> ClassTree::getChildren(int node) const {
  return children[node];
}

int ClassTree::childIndex(int node) const {
  return node - children[parent[node]][0];
}

bool ClassTree::operator==(const ClassTree& other) const {
  return parent == other.parent
      && children == other.children
      && index == other.index;
}

} // namespace oxlm
