#include "lbl/config.h"

namespace oxlm {

ModelData::ModelData()
    : iterations(0), minibatch_size(0), instances(0), ngram_order(0),
      feature_context_size(0), l2_lbl(0), l2_maxent(0),
      word_representation_size(0), threads(0), step_size(0), classes(0),
      randomise(false), reclass(false), diagonal_contexts(false),
      label_sample_size(0), uniform(false), pseudo_likelihood_cne(false),
      mixture(false), lbfgs(false), lbfgs_vectors(0), test_tokens(0),
      gnorm_threshold(0), eta(0), multinomial_step_size(0), log_period(0),
      sparse_features(false), random_weights(false) {}

} // namespace oxlm
