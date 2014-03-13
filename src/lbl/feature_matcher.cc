#include "lbl/feature_matcher.h"

namespace oxlm {

FeatureMatcher::FeatureMatcher(
    const Corpus& corpus, const WordToClassIndex& index,
    const ContextExtractor& extractor, const FeatureGenerator& generator)
    : word_features(index.getNumClasses()) {
  for (size_t i = 0; i < corpus.size(); ++i) {
    int class_id = index.getClass(corpus[i]);
    vector<int> context = extractor.extract(i);
    vector<FeatureContext> feature_contexts = generator.generate(context);
    class_features.push_back(make_pair(feature_contexts, class_id));
    word_features[class_id].push_back(make_pair(
        feature_contexts, index.getWordIndexInClass(corpus[i])));
  }
}

MatchingContexts FeatureMatcher::getClassFeatures() const {
  return class_features;
}

MatchingContexts FeatureMatcher::getWordFeatures(int class_id) const {
  return word_features[class_id];
}

} // namespace oxlm
