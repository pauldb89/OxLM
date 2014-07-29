#pragma once

#include <string>
#include <vector>

#include "lbl/vocabulary.h"

using namespace std;

namespace oxlm {

class ClassTree {
 public:
  ClassTree();

  ClassTree(const string& filename, boost::shared_ptr<Vocabulary>& vocab);

  size_t size() const;

  int getRoot() const;

  int getNode(int word_id) const;

  int getParent(int node) const;

  vector<int> getChildren(int node) const;

  int childIndex(int node) const;

  bool operator==(const ClassTree& other) const;

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & parent;
    ar & children;
    ar & index;
  }

  vector<int> parent;
  vector<vector<int>> children;
  vector<int> index;
};

} // namespace oxlm
