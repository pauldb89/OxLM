#pragma once

#include <unordered_map>

#include <boost/serialization/serialization.hpp>

#include "lbl/ngram_query.h"
#include "utils/serialization_helpers.h"

using namespace std;

namespace oxlm {

class QueryCache {
 public:
  pair<double, bool> get(const NGramQuery& query) const;

  void put(const NGramQuery& query, double value);

  size_t size() const;

  bool operator==(const QueryCache& other) const;

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & cache;
  }

  unordered_map<NGramQuery, double> cache;
};

} // namespace oxlm
