#include "lbl/config.h"

#include "utils/constants.h"

namespace oxlm {

ModelData::ModelData()
    : iterations(0), minibatch_size(0), instances(0), ngram_order(0),
      feature_context_size(0), l2_lbl(0), l2_maxent(0),
      word_representation_size(0), threads(1), step_size(0), classes(0),
      randomise(false), reclass(false), diagonal_contexts(false),
      uniform(false), pseudo_likelihood_cne(false), mixture(false),
      lbfgs(false), lbfgs_vectors(0), test_tokens(0), gnorm_threshold(0),
      eta(0), multinomial_step_size(0), sparse_features(false),
      random_weights(false), hash_space(0), count_collisions(false),
      filter_contexts(false), filter_error_rate(0), max_ngrams(0),
      min_ngram_freq(0), vocab_size(0), noise_samples(0), sigmoid(false) {}

bool ModelData::operator==(const ModelData& other) const {
  if (fabs(l2_lbl - other.l2_lbl) > EPS ||
      fabs(l2_maxent - other.l2_maxent) > EPS) {
    cout << "Warning: Using different regularizers!" << endl;
  }

  return training_file == other.training_file
      && ngram_order == other.ngram_order
      && feature_context_size == other.feature_context_size
      && word_representation_size == other.word_representation_size
      && classes == other.classes
      && class_file == other.class_file
      && diagonal_contexts == other.diagonal_contexts
      && sparse_features == other.sparse_features
      && hash_space == other.hash_space
      && filter_contexts == other.filter_contexts
      && fabs(filter_error_rate - other.filter_error_rate) < EPS
      && sigmoid == other.sigmoid;
}

} // namespace oxlm
