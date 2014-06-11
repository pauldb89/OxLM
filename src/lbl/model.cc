#include "lbl/model.h"

#include <boost/make_shared.hpp>

#include "lbl/factored_metadata.h"
#include "lbl/factored_maxent_metadata.h"
#include "lbl/factored_weights.h"
#include "lbl/global_factored_maxent_weights.h"
#include "lbl/metadata.h"
#include "lbl/minibatch_factored_maxent_weights.h"
#include "lbl/model_utils.h"
#include "lbl/weights.h"
#include "utils/conditional_omp.h"


namespace oxlm {

template<class GlobalWeights, class MinibatchWeights, class Metadata>
Model<GlobalWeights, MinibatchWeights, Metadata>::Model(ModelData& config)
    : config(config) {
  metadata = boost::make_shared<Metadata>(config, dict);
}

template<class GlobalWeights, class MinibatchWeights, class Metadata>
void Model<GlobalWeights, MinibatchWeights, Metadata>::learn() {
  // Initialize the dictionary now, if it hasn't been initialized when the
  // vocabulary was partitioned in classes.
  bool immutable_dict = config.classes > 0 || config.class_file.size();
  boost::shared_ptr<Corpus> training_corpus =
      readCorpus(config.training_file, dict, immutable_dict);
  config.vocab_size = dict.size();

  if (config.test_file.size()) {
    boost::shared_ptr<Corpus> test_corpus = readCorpus(config.test_file, dict);
  }

  metadata->initialize(training_corpus);
  weights = boost::make_shared<GlobalWeights>(config, metadata, true);

  vector<int> indices(training_corpus->size());
  iota(indices.begin(), indices.end(), 0);

  boost::shared_ptr<GlobalWeights> adagrad =
      boost::make_shared<GlobalWeights>(config, metadata);
  boost::shared_ptr<MinibatchWeights> global_gradient =
      boost::make_shared<MinibatchWeights>(config, metadata);

  omp_set_num_threads(config.threads);
  #pragma omp parallel
  {
    int minibatch_counter = 1;
    int minibatch_size = config.minibatch_size;
    for (int iter = 0; iter < config.iterations; ++iter) {
      auto iteration_start = GetTime();

      #pragma omp master
      if (config.randomise) {
        random_shuffle(indices.begin(), indices.end());
      }

      size_t start = 0;
      while (start < training_corpus->size()) {
        size_t end = min(training_corpus->size(), start + minibatch_size);

        #pragma omp master
        global_gradient->clear();

        vector<int> minibatch = scatterMinibatch(start, end, indices);
        boost::shared_ptr<MinibatchWeights> gradient;
        Real objective;
        computeGradient(training_corpus, minibatch, gradient, objective);

        ++minibatch_counter;
        start = end;
      }
    }
  }
}

template<class GlobalWeights, class MinibatchWeights, class Metadata>
void Model<GlobalWeights, MinibatchWeights, Metadata>::computeGradient(
    const boost::shared_ptr<Corpus>& corpus,
    const vector<int>& indices,
    boost::shared_ptr<MinibatchWeights>& gradient,
    Real& objective) const {
  vector<vector<int>> contexts;
  vector<MatrixReal> context_vectors;
  weights->getContextVectors(corpus, indices, contexts, context_vectors);

  MatrixReal prediction_vectors =
      weights->getPredictionVectors(indices, context_vectors);

  weights->getGradient(
      corpus, indices, contexts, context_vectors, prediction_vectors,
      gradient, objective);
}

template<class GlobalWeights, class MinibatchWeights, class Metadata>
void Model<GlobalWeights, MinibatchWeights, Metadata>::regularize() {
}

template<class GlobalWeights, class MinibatchWeights, class Metadata>
void Model<GlobalWeights, MinibatchWeights, Metadata>::evaluate() {
}

template class Model<Weights, Weights, Metadata>;
// template class Model<FactoredWeights, FactoredWeights, FactoredMetadata>;
// template class Model<GlobalFactoredMaxentWeights, MinibatchFactoredMaxentWeights, FactoredMaxentMetadata>;

} // namespace oxlm
