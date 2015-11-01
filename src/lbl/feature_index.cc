#include "lbl/feature_index.h"

namespace oxlm {

FeatureIndex::FeatureIndex() : compact(false) {}

vector<int> FeatureIndex::get(Hash h, const vector<int>& default_value) const {
  if (!compact) {
    auto it = index.find(h);
    if (it == index.end()) {
      return default_value;
    }

    return it->second;
  }

  auto it = markers.find(h);
  if (it == markers.end()) {
    return default_value;
  }

  auto marker = it->second;
  return vector<int>(
      compactIndex.begin() + marker.first,
      compactIndex.begin() + marker.first + marker.second);
}

bool FeatureIndex::contains(Hash h, int value) const {
  vector<int> values = get(h);
  return find(values.begin(), values.end(), value) != values.end();
}

void FeatureIndex::add(Hash h, int value) {
  // We can't add values to a compact index.
  // The time complexity would be O(max_ngrams).
  assert(compact == false);

  vector<int>& values = index[h];
  if (find(values.begin(), values.end(), value) == values.end()) {
    values.push_back(value);
  }
}

size_t FeatureIndex::size() const {
  if (!compact) {
    return index.size();
  }

  return markers.size();
}

} // namespace oxlm
