#pragma once

#include <boost/serialization/split_member.hpp>
#include <boost/serialization/vector.hpp>

#include "lbl/utils.h"

namespace oxlm {

class FeatureIndex {
 public:
  FeatureIndex();

  vector<int> get(Hash h, const vector<int>& default_value = vector<int>()) const;

  bool contains(Hash h, int value) const;

  void add(Hash h, int value);

  size_t size() const;

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void save(Archive& ar, const unsigned int version) const {
    size_t num_entries = index.size();
    ar << num_entries;
    for (const auto& entry: index) {
      int num_values = entry.second.size();
      ar << entry.first << num_values;
    }
    for (const auto& entry: index) {
      ar << entry.first << entry.second;
    }
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int version) {
    compact = true;

    size_t num_entries = 0;
    size_t index_size = 0;
    ar >> num_entries;
    for (size_t i = 0; i < num_entries; ++i) {
      size_t key;
      int num_values;
      ar >> key >> num_values;
      index_size += num_values;
    }

    markers.reserve(num_entries);
    compactIndex.reserve(index_size);
    for (size_t i = 0; i < num_entries; ++i) {
      size_t key;
      vector<int> values;
      ar >> key >> values;

      markers[key] = make_pair(compactIndex.size(), values.size());
      compactIndex.insert(compactIndex.end(), values.begin(), values.end());
    }
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER();

  bool compact;
  unordered_map<Hash, vector<int>> index;

  unordered_map<Hash, pair<int, short>> markers;
  vector<short> compactIndex;
};

typedef boost::shared_ptr<FeatureIndex> FeatureIndexPtr;

} // namespace oxlm
