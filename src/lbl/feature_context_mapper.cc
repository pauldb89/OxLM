#include "feature_context_mapper.h"

#include <boost/make_shared.hpp>

namespace oxlm {

FeatureContextMapper::FeatureContextMapper() {}

FeatureContextMapper::FeatureContextMapper(
    const boost::shared_ptr<Corpus>& corpus,
    const boost::shared_ptr<WordToClassIndex>& index,
    const boost::shared_ptr<ContextProcessor>& processor,
    const boost::shared_ptr<FeatureContextGenerator>& generator,
    const boost::shared_ptr<NGramFilter>& filter)
    : index(index), generator(generator),
      wordContextIdsMap(index->getNumClasses()) {
  for (size_t i = 0; i < corpus->size(); ++i) {
    int word_id = corpus->at(i);
    int class_id = index->getClass(word_id);
    vector<int> context = processor->extract(i);

    vector<FeatureContext> feature_contexts =
        generator->getFeatureContexts(context);

    // Map feature context to id if at least one n-gram having that context is
    // within the topmost max_ngrams ngrams.
    feature_contexts = filter->filter(word_id, class_id, feature_contexts);
    for (const FeatureContext& feature_context: feature_contexts) {
      size_t context_hash = hashFunction(feature_context);
      classContextIdsMap.insert(make_pair(
          context_hash, classContextIdsMap.size()));
      wordContextIdsMap[class_id].insert(make_pair(
          context_hash, wordContextIdsMap[class_id].size()));
    }
  }
}

vector<int> FeatureContextMapper::getClassContextIds(
    const vector<int>& context) const {
  return getClassContextIds(generator->getFeatureContexts(context));
}

vector<int> FeatureContextMapper::getClassContextIds(
    const vector<FeatureContext>& feature_contexts) const {
  vector<int> class_context_ids;
  for (const auto& feature_context: feature_contexts) {
    // Feature contexts for the test set are not guaranteed to exist in the
    // hash. Unobserved contexts are skipped.
    int class_context_id = getClassContextId(feature_context);
    if (class_context_id != -1) {
      class_context_ids.push_back(class_context_id);
    }
  }

  return class_context_ids;
}

vector<int> FeatureContextMapper::getWordContextIds(
    int class_id, const vector<WordId>& context) const {
  return getWordContextIds(class_id, generator->getFeatureContexts(context));
}

vector<int> FeatureContextMapper::getWordContextIds(
    int class_id, const vector<FeatureContext>& feature_contexts) const {
  vector<int> word_context_ids;
  for (const auto& feature_context: feature_contexts) {
    // Feature contexts for the test set are not guaranteed to exist in the
    // hash. Unobserved contexts are skipped.
    int word_context_id = getWordContextId(class_id, feature_context);
    if (word_context_id != -1) {
      word_context_ids.push_back(word_context_id);
    }
  }

  return word_context_ids;
}

int FeatureContextMapper::getClassContextId(
    const FeatureContext& feature_context) const {
  size_t context_hash = hashFunction(feature_context);
  auto it = classContextIdsMap.find(context_hash);
  return it == classContextIdsMap.end() ? -1 : it->second;
}

int FeatureContextMapper::getWordContextId(
    int class_id, const FeatureContext& feature_context) const {
  size_t context_hash = hashFunction(feature_context);
  auto it = wordContextIdsMap[class_id].find(context_hash);
  return it == wordContextIdsMap[class_id].end() ? -1 : it->second;
}

int FeatureContextMapper::getNumContexts() const {
  int num_contexts = classContextIdsMap.size();
  for (size_t i = 0; i < wordContextIdsMap.size(); ++i) {
    num_contexts += wordContextIdsMap[i].size();
  }
  return num_contexts;
}

int FeatureContextMapper::getNumClassContexts() const {
  return classContextIdsMap.size();
}

int FeatureContextMapper::getNumWordContexts(int class_id) const {
  return wordContextIdsMap[class_id].size();
}

bool FeatureContextMapper::operator==(const FeatureContextMapper& other) const {
  return *index == *other.index
      && *generator == *other.generator
      && classContextIdsMap == other.classContextIdsMap
      && wordContextIdsMap == other.wordContextIdsMap;
}

} // namespace oxlm
