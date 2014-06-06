#pragma once

#include <vector>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

#include "lbl/utils.h"

using namespace std;

namespace oxlm {

struct NGram {
  NGram();

  NGram(int word, const vector<int>& context);

  NGram(int word, int class_id, const vector<int>& context);

  bool operator==(const NGram& other) const;

  int word;
  int classId;
  vector<int> context;

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & word;
    ar & classId;
    ar & context;
  }
};

} // namespace oxlm


namespace std {

template<> class hash<oxlm::NGram> {
 public:
  hash<oxlm::NGram>(int seed = 0) : seed(seed) {}

  inline size_t operator()(const oxlm::NGram query) const {
    vector<int> data;
    data.push_back(query.word);
    data.push_back(query.classId);
    data.insert(data.end(), query.context.begin(), query.context.end());
    return oxlm::MurmurHash(data, seed);
  }

  bool operator==(const hash<oxlm::NGram>& other) const {
    return seed == other.seed;
  }

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & seed;
  }

  int seed;
};

} // namespace std
