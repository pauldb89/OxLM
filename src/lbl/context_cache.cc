#include "lbl/context_cache.h"

namespace oxlm {

pair<Real, bool> ContextCache::get(const vector<int>& context) {
  if (!cache.get()) {
    cache.reset(new ContextMap());
  }

  auto it = cache->find(context);
  if (it == cache->end()) {
    return make_pair(0, false);
  } else {
    return make_pair(it->second, true);
  }
}

void ContextCache::set(const vector<int>& context, Real value) {
  assert(cache.get());
  cache->insert(make_pair(context, value));
}

void ContextCache::clear() {
  if (!cache.get()) {
    cache.reset(new ContextMap());
  }

  cache->clear();
}

} // namespace oxlm
