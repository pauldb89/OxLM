#pragma once

#include <fstream>
#include <iostream>
#include <map>

#include <boost/lexical_cast.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/serialization/serialization.hpp>

using namespace std;

namespace oxlm {

struct UnigramDistribution {
  map<double, string> prob_to_token;
  map<string, double> token_to_prob;

  void read(const string& filename) {
    ifstream file(filename.c_str());
    cerr << "Reading unigram distribution from "
      << filename.c_str() << "." << endl;

    double sum=0;
    string key, value;
    while (file >> value >> key) {
      double v = boost::lexical_cast<double>(value);
      sum += v;
      prob_to_token.insert(make_pair(sum, key));
      token_to_prob.insert(make_pair(key, v));
    }
  }

  double prob(const string& s) const {
   map<string, double>::const_iterator it
     = token_to_prob.find(s);
   return it != token_to_prob.end() ? it->second : 0.0;
 }

  bool empty() const { return prob_to_token.empty(); }
};

struct ModelData {
  ModelData();

  string      training_file;
  string      test_file;
  int         iterations;
  int         minibatch_size;
  int         instances;
  int         ngram_order;
  int         feature_context_size;
  string      model_input_file;
  string      model_output_file;
  float       l2_lbl;
  float       l2_maxent;
  int         word_representation_size;
  int         threads;
  float       step_size;
  int         classes;
  string      class_file;
  bool        randomise;
  bool        reclass;
  bool        diagonal_contexts;
  bool        uniform;
  bool        pseudo_likelihood_cne;
  bool        mixture;
  bool        lbfgs;
  int         lbfgs_vectors;
  int         test_tokens;
  float       gnorm_threshold;
  float       eta;
  float       multinomial_step_size;
  bool        sparse_features;
  bool        random_weights;
  int         hash_space;
  bool        count_collisions;
  bool        filter_contexts;
  float       filter_error_rate;
  int         max_ngrams;
  int         min_ngram_freq;
  int         vocab_size;
  int         noise_samples;
  bool        sigmoid;

  bool operator==(const ModelData& other) const;

  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & training_file;
    ar & test_file;
    ar & iterations;
    ar & minibatch_size;
    ar & instances;
    ar & ngram_order;
    ar & feature_context_size;
    ar & model_input_file;
    ar & model_output_file;
    ar & l2_lbl;
    ar & l2_maxent;
    ar & word_representation_size;
    ar & step_size;
    ar & classes;
    ar & class_file;
    ar & randomise;
    ar & diagonal_contexts;
    ar & sparse_features;
    ar & hash_space;
    ar & filter_contexts;
    ar & filter_error_rate;
    ar & vocab_size;
    ar & noise_samples;
    ar & sigmoid;
  }
};

} // namespace oxlm
