#include "lbl/factored_maxent_nlm.h"

#include <boost/make_shared.hpp>

#include "lbl/feature_generator.h"

namespace oxlm {

FactoredMaxentNLM::FactoredMaxentNLM() {}

FactoredMaxentNLM::FactoredMaxentNLM(
    const ModelData& config, const Dict& labels,
    const WordToClassIndex& index, const FeatureStoreInitializer& initializer)
    : FactoredNLM(config, labels, index) {
  initializer.initialize(U, V);
}


Real FactoredMaxentNLM::log_prob(
    const WordId w, const vector<WordId>& context,
    bool non_linear=false, bool cache=false) const {
  VectorReal prediction_vector = VectorReal::Zero(config.word_representation_size);
  int width = config.ngram_order-1;
  int gap = width-context.size();
  assert(gap == 0);
  assert(static_cast<int>(context.size()) <= width);
  for (int i=gap; i < width; i++)
    if (m_diagonal) prediction_vector += C.at(i).asDiagonal() * Q.row(context.at(i-gap)).transpose();
    else            prediction_vector += Q.row(context.at(i-gap)) * C.at(i);

  int c = get_class(w);
  int word_index = index.getWordIndexInClass(w);
  FeatureGenerator generator(config.feature_context_size);
  vector<FeatureContext> feature_contexts = generator.generate(context);
  VectorReal class_feature_scores = U->get(feature_contexts);
  VectorReal word_feature_scores = V[c]->get(feature_contexts);

  // a simple non-linearity
  if (non_linear) {
    prediction_vector = sigmoid(prediction_vector);
  }

  // log p(c | context)
  Real class_log_prob = 0;
  pair<unordered_map<Words, Real, container_hash<Words> >::iterator, bool> context_cache_result;
  if (cache) context_cache_result = m_context_cache.insert(make_pair(context,0));
  if (cache && !context_cache_result.second) {
    assert (context_cache_result.first->second != 0);
    class_log_prob = F.row(c)*prediction_vector + FB(c) + class_feature_scores(c) - context_cache_result.first->second;
  } else {
    Real c_log_z=0;
    VectorReal class_probs = logSoftMax(F*prediction_vector + FB + class_feature_scores, &c_log_z);
    assert(c_log_z != 0);
    class_log_prob = class_probs(c);
    if (cache) {
      context_cache_result.first->second = c_log_z;
    }
  }

  // log p(w | c, context)
  Real word_log_prob = 0;
  pair<unordered_map<pair<int,Words>, Real>::iterator, bool> class_context_cache_result;
  if (cache) class_context_cache_result = m_context_class_cache.insert(make_pair(make_pair(c,context),0));

  if (cache && !class_context_cache_result.second) {
    word_log_prob  = R.row(w)*prediction_vector + B(w) + word_feature_scores(word_index) - class_context_cache_result.first->second;
  } else {
    Real w_log_z = 0;
    VectorReal word_probs = logSoftMax(class_R(c) * prediction_vector + class_B(c) + word_feature_scores, &w_log_z);
    word_log_prob = word_probs(word_index);
    if (cache) {
      class_context_cache_result.first->second = w_log_z;
    }
  }

  return class_log_prob + word_log_prob;
}

Real FactoredMaxentNLM::l2_gradient_update(Real lambda) {
  Real result = FactoredNLM::l2_gradient_update(lambda);
  result += U->updateRegularizer(lambda);
  for (auto& store: V) {
    result += store->updateRegularizer(lambda);
  }
  return result;
}

} // namespace oxlm
