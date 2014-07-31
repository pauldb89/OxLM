#include "lbl/query_cache.h"

namespace oxlm {

pair<Real, bool> QueryCache::get(const NGram& query) const {
  auto it = cache.find(hashFunction(query));
  return it == cache.end() ? make_pair(Real(0), false) : make_pair(it->second, true);
}

void QueryCache::put(const NGram& query, Real value) {
  cache[hashFunction(query)] = value;
}

size_t QueryCache::size() const {
  return cache.size();
}

void QueryCache::clear() {
  cache.clear();
}

bool QueryCache::operator==(const QueryCache& other) const {
  return cache == other.cache;
}

} // namespace oxlm
