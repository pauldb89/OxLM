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
  cout << "Done reading training corpus..." << endl;

  boost::shared_ptr<Corpus> test_corpus;
  if (config.test_file.size()) {
    test_corpus = readCorpus(config.test_file, dict);
    cout << "Done reading test corpus..." << endl;
  }

  metadata->initialize(training_corpus);
  weights = boost::make_shared<GlobalWeights>(config, metadata, true);

  vector<int> indices(training_corpus->size());
  iota(indices.begin(), indices.end(), 0);

  Real perplexity = 0, best_perplexity = 0, global_objective = 0;
  boost::shared_ptr<MinibatchWeights> global_gradient =
      boost::make_shared<MinibatchWeights>(config, metadata);
  boost::shared_ptr<GlobalWeights> adagrad =
      boost::make_shared<GlobalWeights>(config, metadata);

  omp_set_num_threads(config.threads);
  #pragma omp parallel
  {
    int minibatch_counter = 1;
    int minibatch_size = config.minibatch_size;
    for (int iter = 0; iter < config.iterations; ++iter) {
      auto iteration_start = GetTime();

      #pragma omp master
      {
        if (config.randomise) {
          random_shuffle(indices.begin(), indices.end());
        }
        global_objective = 0;
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

        #pragma critical
        {
          global_gradient->update(gradient);
          global_objective += objective;
        }

        #pragma omp barrier

        #pragma omp master
        {
          update(global_gradient, adagrad);

          Real minibatch_factor =
              static_cast<Real>(end - start) / training_corpus->size();
          global_objective += regularize(minibatch_factor);
        }

        // Wait for master thread to update model.
        #pragma omp barrier
        if ((minibatch_counter % 100 == 0 && minibatch_counter <= 1000) ||
            minibatch_counter % 1000 == 0) {
          evaluate(test_corpus, iteration_start, minibatch_counter,
                   perplexity, best_perplexity);
        }

        ++minibatch_counter;
        start = end;
      }

      evaluate(test_corpus, iteration_start, minibatch_counter,
               perplexity, best_perplexity);
      #pragma master
      {
        Real iteration_time = GetDuration(iteration_start, GetTime());
        cout << "Iteration: " << iter << ", "
             << "Time: " << iteration_time << "seconds, "
             << "Objective: " << global_objective / training_corpus->size()
             << endl;
        cout << endl;
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
void Model<GlobalWeights, MinibatchWeights, Metadata>::update(
    const boost::shared_ptr<MinibatchWeights>& global_gradient,
    const boost::shared_ptr<GlobalWeights>& adagrad) {
  adagrad->updateSquared(global_gradient);
  weights->updateAdaGrad(global_gradient, adagrad);
}

template<class GlobalWeights, class MinibatchWeights, class Metadata>
Real Model<GlobalWeights, MinibatchWeights, Metadata>::regularize(
    Real minibatch_factor) {
  return weights->regularizerUpdate(minibatch_factor);
}

template<class GlobalWeights, class MinibatchWeights, class Metadata>
Real Model<GlobalWeights, MinibatchWeights, Metadata>::computePerplexity(
    const boost::shared_ptr<Corpus>& test_corpus) const {
  Real perplexity = 0;
  int context_width = config.ngram_order - 1;
  boost::shared_ptr<ContextProcessor> processor =
      boost::make_shared<ContextProcessor>(test_corpus, context_width);

  int tokens = 1;
  size_t thread_num = omp_get_thread_num();
  size_t num_threads = omp_get_num_threads();
  for (size_t i = thread_num; i < test_corpus->size(); i += num_threads) {
    vector<int> context = processor->extract(i);
    perplexity += weights->predict(test_corpus->at(i), context);

    #pragma omp master
    if (tokens % 10000 == 0) {
      cout << ".";
      cout.flush();
    }

    ++tokens;
  }

  return perplexity;
}

template<class GlobalWeights, class MinibatchWeights, class Metadata>
void Model<GlobalWeights, MinibatchWeights, Metadata>::evaluate(
    const boost::shared_ptr<Corpus>& test_corpus, const Time& iteration_start,
    int minibatch_counter, Real& perplexity, Real& best_perplexity) const {
  if (test_corpus != nullptr) {
    #pragma omp master
    perplexity = 0;

    // Each thread must wait until the perplexity is set to 0.
    // Otherwise, partial results might get overwritten.
    #pragma omp barrier

    Real local_perplexity = computePerplexity(test_corpus);
    #pragma omp critical
    perplexity += local_perplexity;

    // Wait for all the threads to compute the perplexity for their slice of
    // test data.
    #pragma omp barrier
    #pragma omp master
    {
      perplexity = exp(-perplexity / test_corpus->size());
      Real iteration_time = GetDuration(iteration_start, GetTime());
      cout << "\tMinibatch " << minibatch_counter << ", "
           << "Time: " << GetDuration(iteration_start, GetTime()) << " seconds, "
           << "Test Perplexity: " << perplexity << endl;

      if (perplexity < best_perplexity) {
        best_perplexity = perplexity;
        save();
      }
    }
  } else {
    #pragma omp master
    save();
  }
}

template<class GlobalWeights, class MinibatchWeights, class Metadata>
void Model<GlobalWeights, MinibatchWeights, Metadata>::save() const {
}

template class Model<Weights, Weights, Metadata>;
// template class Model<FactoredWeights, FactoredWeights, FactoredMetadata>;
// template class Model<GlobalFactoredMaxentWeights, MinibatchFactoredMaxentWeights, FactoredMaxentMetadata>;

} // namespace oxlm
