#include "lbl/config.h"

#include "utils/constants.h"

namespace oxlm {

ModelData::ModelData()
    : iterations(0), evaluate_frequency(1), minibatch_size(0),
      minibatch_threshold(0), instances(0), ngram_order(0),
      feature_context_size(0), l2_lbl(0), l2_maxent(0),
      word_representation_size(0), threads(1), step_size(0), classes(0),
      randomise(false), reclass(false), diagonal_contexts(false),
      uniform(false), pseudo_likelihood_cne(false), mixture(false),
      lbfgs(false), lbfgs_vectors(0), test_tokens(0), gnorm_threshold(0),
      eta(0), multinomial_step_size(0), random_weights(false), hash_space(0),
      count_collisions(false), filter_contexts(false), filter_error_rate(0),
      max_ngrams(0), min_ngram_freq(0), vocab_size(0), noise_samples(0),
      activation(IDENTITY), source_order(0), source_vocab_size(0),
      hidden_layers(0) {}

bool ModelData::operator==(const ModelData& other) const {
  if (fabs(l2_lbl - other.l2_lbl) > EPS ||
      fabs(l2_maxent - other.l2_maxent) > EPS) {
    cout << "Warning: Using different regularizers!" << endl;
  }

  return training_file == other.training_file
      && alignment_file == other.alignment_file
      && ngram_order == other.ngram_order
      && feature_context_size == other.feature_context_size
      && word_representation_size == other.word_representation_size
      && classes == other.classes
      && class_file == other.class_file
      && tree_file == other.tree_file
      && diagonal_contexts == other.diagonal_contexts
      && hash_space == other.hash_space
      && filter_contexts == other.filter_contexts
      && fabs(filter_error_rate - other.filter_error_rate) < EPS
      && activation == other.activation
      && source_order == other.source_order
      && hidden_layers == other.hidden_layers;
}

ostream& operator<<(ostream& out, const ModelData& config) {
  out << "################################" << endl;

  out << "# Input/Output files:" << endl;
  out << "# input = " << config.training_file << endl;
  out << "# test file = " << config.test_file << endl;
  if (config.class_file.size()) {
    out << "# class file = " << config.class_file << endl;
  }
  out << "# model-out = " << config.model_output_file << endl;
  out << endl;

  out << "# General config: " << endl;
  out << "# vocab size = " << config.vocab_size << endl;
  out << "# order = " << config.ngram_order << endl;
  out << "# word_width = " << config.word_representation_size << endl;
  out << "# iterations = " << config.iterations << endl;
  out << "# minibatch size = " << config.minibatch_size << endl;
  out << "# evaluate frequency = " << config.evaluate_frequency << endl;
  out << "# minibatch threshold = " << config.minibatch_threshold << endl;
  out << "# lambda = " << config.l2_lbl << endl;
  out << "# step size = " << config.step_size << endl;
  out << "# threads = " << config.threads << endl;
  out << "# randomise = " << config.randomise << endl;
  out << "# diagonal contexts = " << config.diagonal_contexts << endl;
  out << "# activation = " << config.activation << endl;
  out << "# noise samples = " << config.noise_samples << endl;

  if (config.l2_maxent > 0 || config.hash_space > 0) {
    out << "# Direct n-grams config: " << endl;
    out << "# lambda maxent = " << config.l2_maxent << endl;
    out << "# max n-grams = " << config.max_ngrams << endl;
    out << "# min n-gram frequency = " << config.min_ngram_freq << endl;
    out << "# hash space = " << config.hash_space << endl;
    out << "# filter contexts = " << config.filter_contexts << endl;
    out << "# filter error rate = " << config.filter_error_rate << endl;
  }

  if (config.source_vocab_size > 0 || config.source_order > 0) {
    out << "Source conditioning config:" << endl;
    out << "# source vocab size = " << config.source_vocab_size << endl;
    out << "# source order = " << config.source_order << endl;
    out << "# alignment file = " << config.alignment_file << endl;
    out << "# test alignment file = " << config.test_alignment_file << endl;
  }

  out << "################################" << endl;
}

} // namespace oxlm
