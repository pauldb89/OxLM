#include "feature_context_hasher.h"

#include <boost/make_shared.hpp>

namespace oxlm {

FeatureContextHasher::FeatureContextHasher() {}

FeatureContextHasher::FeatureContextHasher(
    const boost::shared_ptr<Corpus>& corpus,
    const boost::shared_ptr<WordToClassIndex>& index,
    const boost::shared_ptr<ContextProcessor>& processor,
    size_t feature_context_size)
    : index(index), wordContextIdsMap(index->getNumClasses()) {
  generator = boost::make_shared<FeatureContextGenerator>(feature_context_size);
  for (size_t i = 0; i < corpus->size(); ++i) {
    int class_id = index->getClass(corpus->at(i));
    vector<int> context = processor->extract(i);
    for (const auto& feature_context: generator->getFeatureContexts(context)) {
      classContextIdsMap.insert(make_pair(
          feature_context, classContextIdsMap.size()));
      wordContextIdsMap[class_id].insert(make_pair(
          feature_context, wordContextIdsMap[class_id].size()));
    }
  }
}

vector<int> FeatureContextHasher::getClassContextIds(
    const vector<int>& context) const {
  vector<int> class_context_ids;
  for (const auto& feature_context: generator->getFeatureContexts(context)) {
    // Feature contexts for the test set are not guaranteed to exist in the
    // hash. Unobserved contexts are skipped.
    auto it = classContextIdsMap.find(feature_context);
    if (it != classContextIdsMap.end()) {
      class_context_ids.push_back(it->second);
    }
  }

  return class_context_ids;
}

vector<int> FeatureContextHasher::getWordContextIds(
    int class_id, const vector<WordId>& context) const {
  vector<int> word_context_ids;
  for (const auto& feature_context: generator->getFeatureContexts(context)) {
    // Feature contexts for the test set are not guaranteed to exist in the
    // hash. Unobserved contexts are skipped.
    auto it = wordContextIdsMap[class_id].find(feature_context);
    if (it != classContextIdsMap.end()) {
      word_context_ids.push_back(it->second);
    }
  }

  return word_context_ids;
}

int FeatureContextHasher::getNumClassContexts() const {
  return classContextIdsMap.size();
}

int FeatureContextHasher::getNumWordContexts(int class_id) const {
  return wordContextIdsMap[class_id].size();
}

bool FeatureContextHasher::operator==(const FeatureContextHasher& other) const {
  return *index == *other.index
      && *generator == *other.generator
      && classContextIdsMap == other.classContextIdsMap
      && wordContextIdsMap == other.wordContextIdsMap;
}

} // namespace oxlm
