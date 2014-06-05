#pragma once

#include <vector>

#include <boost/serialization/serialization.hpp>

#include "lbl/utils.h"

using namespace std;

namespace oxlm {

template<class T>
class BloomFilter {
 public:
  BloomFilter() {}

  BloomFilter(int num_items, int min_frequency, Real error_rate) :
      minFrequency(min_frequency), errorRate(error_rate) {
    cout << "Creating Bloom Filter for " << num_items << " contexts..." << endl;
    bucketSize = ceil(log2(minFrequency + 1));

    numBuckets = -num_items * log(errorRate) / (log(2) * log(2));
    cout << "Bloom Filter size: " << numBuckets << " buckets..." << endl;
    int num_hashes = round(numBuckets * log(2) / num_items);
    for (int i = 0; i < num_hashes; ++i) {
      hashes.push_back(hash<T>(i));
    }

    bits.reserve(numBuckets * bucketSize);
    bits.resize(numBuckets * bucketSize, false);
  }

  void increment(const T& item) {
    int min_value = minFrequency;
    vector<int> keys(hashes.size()), values(hashes.size());
    for (size_t i = 0; i < hashes.size(); ++i) {
      keys[i] = bucketSize * (hashes[i](item) % numBuckets);
      values[i] = getValue(keys[i]);
      min_value = min(min_value, values[i]);
    }

    if (min_value < minFrequency) {
      for (size_t i = 0; i < hashes.size(); ++i) {
        if (min_value == values[i]) {
          setValue(keys[i], min_value + 1);
        }
      }
    }
  }

  bool contains(const T& item) const {
    for (const auto& hash: hashes) {
      int key = bucketSize * (hash(item) % numBuckets);
      if (getValue(key) < minFrequency) {
        return false;
      }
    }

    return true;
  }

  bool operator==(const BloomFilter<T>& other) const {
    return numBuckets == other.numBuckets
        && bucketSize == other.bucketSize
        && minFrequency == other.minFrequency
        && errorRate == other.errorRate
        && hashes == other.hashes
        && bits == other.bits;
  }

 private:
  int getValue(int key) const {
    int result = 0;
    for (int i = 0; i < bucketSize; ++i) {
      result += bits[key + i] << i;
    }

    return result;
  }

  void setValue(int key, int value) {
    assert(value <= minFrequency);
    for (int i = 0; i < bucketSize; ++i) {
      bits[key + i] = value & 1;
      value >>= 1;
    }
  }

  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & numBuckets;
    ar & bucketSize;
    ar & minFrequency;
    ar & errorRate;
    ar & hashes;
    ar & bits;
  }

  int numBuckets;
  int bucketSize;
  int minFrequency;
  Real errorRate;
  vector<hash<T>> hashes;
  // The underlying storage for vector<bool> is not necessarily an array of bool
  // values, but the library implementation may optimize storage so that each
  // value is stored in a single bit.
  // Reference: http://www.cplusplus.com/reference/vector/vector-bool/
  // GCC stores a single bit per value (tested on version 4.6.3).
  vector<bool> bits;
};

} // namespace oxlm
