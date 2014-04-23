#include "feature_context_extractor.h"

namespace oxlm {

FeatureContextExtractor::FeatureContextExtractor() {}

FeatureContextExtractor::FeatureContextExtractor(
    const boost::shared_ptr<Corpus>& corpus,
    const boost::shared_ptr<WordToClassIndex>& index,
    const boost::shared_ptr<ContextProcessor>& processor,
    size_t feature_context_size)
    : index(index), featureContextSize(feature_context_size),
      wordContextIdsMap(index->getNumClasses()) {
  for (size_t i = 0; i < corpus->size(); ++i) {
    int class_id = index->getClass(corpus->at(i));
    vector<int> context = processor->extract(i);
    vector<FeatureContext> feature_contexts = getFeatureContexts(context);
    for (const FeatureContext& feature_context: feature_contexts) {
      classContextIdsMap.insert(make_pair(
          feature_context, classContextIdsMap.size()));
      wordContextIdsMap[class_id].insert(make_pair(
          feature_context, wordContextIdsMap[class_id].size()));
    }
  }
}

pair<vector<int>, vector<int>> FeatureContextExtractor::getFeatureContextIds(
    int class_id, const vector<WordId>& history) const {
  vector<int> class_context_ids, word_context_ids;
  vector<FeatureContext> feature_contexts = getFeatureContexts(history);
  for (const FeatureContext& feature_context: feature_contexts) {
    // Feature contexts for the test set are not guaranteed to exist in the
    // hash. Unobserved contexts are skipped.
    auto it = classContextIdsMap.find(feature_context);
    if (it != classContextIdsMap.end()) {
      class_context_ids.push_back(it->second);
    }

    it = wordContextIdsMap[class_id].find(feature_context);
    if (it != classContextIdsMap.end()) {
      word_context_ids.push_back(it->second);
    }
  }
  return make_pair(class_context_ids, word_context_ids);
}

vector<FeatureContext> FeatureContextExtractor::getFeatureContexts(
    const vector<WordId>& history) const {
  vector<FeatureContext> feature_contexts;
  vector<int> context;
  for (size_t i = 0; i < min(featureContextSize, history.size()); ++i) {
    context.reserve(context.size() + 1);
    context.push_back(history[i]);
    feature_contexts.push_back(FeatureContext(context));
  }
  return feature_contexts;
}

int FeatureContextExtractor::getNumClassContexts() const {
  return classContextIdsMap.size();
}

int FeatureContextExtractor::getNumWordContexts(int class_id) const {
  return wordContextIdsMap[class_id].size();
}

} // namespace oxlm
