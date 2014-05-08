#include "lbl/query_cache.h"

namespace oxlm {

pair<double, bool> QueryCache::get(const NGramQuery& query) const {
  auto it = cache.find(query);
  return it == cache.end() ? make_pair(0.0, false) : make_pair(it->second, true);
}

void QueryCache::put(const NGramQuery& query, double value) {
  cache[query] = value;
}

size_t QueryCache::size() const {
  return cache.size();
}

bool QueryCache::operator==(const QueryCache& other) const {
  return cache == other.cache;
}

} // namespace oxlm
