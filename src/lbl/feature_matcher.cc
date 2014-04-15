#include "lbl/feature_matcher.h"

namespace oxlm {

FeatureMatcher::FeatureMatcher(
    const Corpus& corpus, const WordToClassIndex& index,
    const ContextProcessor& processor,
    const boost::shared_ptr<FeatureContextExtractor>& extractor)
    : index(index), processor(processor),
      extractor(extractor), word_features(index.getNumClasses()) {
  for (size_t i = 0; i < corpus.size(); ++i) {
    int class_id = index.getClass(corpus[i]);
    int word_class_id = index.getWordIndexInClass(corpus[i]);
    vector<WordId> context = processor.extract(i);
    vector<FeatureContextId> feature_context_ids =
        extractor->getFeatureContextIds(context);
    for (const FeatureContextId& feature_context_id: feature_context_ids) {
      class_features[feature_context_id].insert(class_id);
      word_features[class_id][feature_context_id].insert(word_class_id);
    }
  }
}

FeatureIndexes FeatureMatcher::getClassFeatures() const {
  return class_features;
}

FeatureIndexes FeatureMatcher::getWordFeatures(int class_id) const {
  return word_features[class_id];
}

} // namespace oxlm
