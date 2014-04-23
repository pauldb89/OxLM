#pragma once

#include <unordered_map>
#include <unordered_set>

#include <boost/serialization/serialization.hpp>

namespace boost {
namespace serialization {

// Serialization support for Eigen::SparseVector<Scalar>.

template<class Archive, class Scalar>
inline void save(
    Archive& ar, const Eigen::SparseVector<Scalar>& v,
    const unsigned int version) {
  int max_size = v.size();
  ar << max_size;
  int actual_size = v.nonZeros();
  ar << actual_size;
  ar << boost::serialization::make_array(&v._data().index(0), actual_size);
  ar << boost::serialization::make_array(&v._data().value(0), actual_size);
}

template<class Archive, class Scalar>
inline void load(
    Archive& ar, Eigen::SparseVector<Scalar>& v, const unsigned int version) {
  int max_size;
  ar >> max_size;
  v.resize(max_size);
  int actual_size;
  ar >> actual_size;
  v.resizeNonZeros(actual_size);
  ar >> boost::serialization::make_array(&v._data().index(0), actual_size);
  ar >> boost::serialization::make_array(&v._data().value(0), actual_size);
}

template<class Archive, class Scalar>
inline void serialize(
    Archive& ar, Eigen::SparseVector<Scalar>& v, const unsigned int version) {
  boost::serialization::split_free(ar, v, version);
}


// Serialization support for std::unordered_map<Key, Value>.

template<class Archive, class Key, class Value>
inline void save(
    Archive& ar, const std::unordered_map<Key, Value>& map,
    const unsigned int version) {
  size_t num_entries = map.size();
  ar << num_entries;
  for (const pair<Key, Value>& item: map) {
    ar << item.first << item.second;
  }
}

template<class Archive, class Key, class Value>
inline void load(
    Archive& ar, std::unordered_map<Key, Value>& map,
    const unsigned int version) {
  size_t num_entries;
  ar >> num_entries;
  for (size_t i = 0; i < num_entries; ++i) {
    Key key;
    Value value;
    ar >> key >> value;
    map.insert(make_pair(key, value));
  }
}

template<class Archive, class Key, class Value>
inline void serialize(
    Archive& ar, std::unordered_map<Key, Value>& map,
    const unsigned int version) {
  boost::serialization::split_free(ar, map, version);
}


// Serialization support for std::unordered_set<Value>.

template<class Archive, class Value>
inline void save(
    Archive& ar, const std::unordered_set<Value>& set,
    const unsigned int version) {
  size_t num_entries = set.size();
  ar << num_entries;
  for (const Value& value: set) {
    ar << value;
  }
}

template<class Archive, class Value>
inline void load(
    Archive& ar, std::unordered_set<Value>& set, const unsigned int version) {
  size_t num_entries;
  ar >> num_entries;
  for (size_t i = 0; i < num_entries; ++i) {
    Value value;
    ar >> value;
    set.insert(value);
  }
}

template<class Archive, class Value>
inline void serialize(
    Archive& ar, std::unordered_set<Value>& set, const unsigned int version) {
  boost::serialization::split_free(ar, set, version);
}

} // namespace serialization
} // namespace boost

