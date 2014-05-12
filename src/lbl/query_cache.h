#pragma once

#include <unordered_map>

#include <boost/serialization/serialization.hpp>

#include "lbl/ngram_query.h"
#include "lbl/utils.h"
#include "utils/serialization_helpers.h"

using namespace std;

namespace oxlm {

class QueryCache {
 public:
  pair<Real, bool> get(const NGramQuery& query) const;

  void put(const NGramQuery& query, Real value);

  size_t size() const;

  bool operator==(const QueryCache& other) const;

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & cache;
  }

 public:
  hash<NGramQuery> hashFunction;
  unordered_map<size_t, Real> cache;
};

} // namespace oxlm
