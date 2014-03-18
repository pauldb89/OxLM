#include "lbl/feature_matcher.h"

namespace oxlm {

FeatureMatcher::FeatureMatcher(
    const Corpus& corpus, const WordToClassIndex& index,
    const ContextExtractor& extractor,
    const boost::shared_ptr<FeatureGenerator>& generator)
    : word_features(index.getNumClasses()) {
  for (size_t i = 0; i < corpus.size(); ++i) {
    int class_id = index.getClass(corpus[i]);
    vector<WordId> context = extractor.extract(i);
    vector<FeatureContextId> feature_context_ids =
        generator->getFeatureContextIds(context);
    class_features.push_back(make_pair(feature_context_ids, class_id));
    word_features[class_id].push_back(make_pair(
        feature_context_ids, index.getWordIndexInClass(corpus[i])));
  }
}

MatchingContexts FeatureMatcher::getClassFeatures() const {
  return class_features;
}

MatchingContexts FeatureMatcher::getWordFeatures(int class_id) const {
  return word_features[class_id];
}

} // namespace oxlm
