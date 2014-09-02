#pragma once

#include <boost/make_shared.hpp>
#include <boost/thread/tss.hpp>

#include "lbl/class_distribution.h"
#include "lbl/factored_metadata.h"
#include "lbl/weights.h"
#include "lbl/word_distributions.h"
#include "lbl/word_to_class_index.h"

namespace oxlm {

class FactoredWeights : public Weights {
 public:
  FactoredWeights();

  FactoredWeights(
      const boost::shared_ptr<ModelData>& config,
      const boost::shared_ptr<FactoredMetadata>& metadata);

  FactoredWeights(
      const boost::shared_ptr<ModelData>& config,
      const boost::shared_ptr<FactoredMetadata>& metadata,
      const boost::shared_ptr<Corpus>& training_corpus);

  FactoredWeights(const FactoredWeights& other);

  virtual size_t numParameters() const;

  void init(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& minibatch);

  void getGradient(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const boost::shared_ptr<FactoredWeights>& gradient,
      Real& objective,
      MinibatchWords& words) const;

  virtual Real getObjective(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices) const;

  bool checkGradient(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const boost::shared_ptr<FactoredWeights>& gradient,
      double eps);

  void estimateGradient(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const boost::shared_ptr<FactoredWeights>& gradient,
      Real& objective,
      MinibatchWords& words) const;

  void syncUpdate(
      const MinibatchWords& words,
      const boost::shared_ptr<FactoredWeights>& gradient);

  void updateSquared(
      const MinibatchWords& global_words,
      const boost::shared_ptr<FactoredWeights>& global_gradient);

  void updateAdaGrad(
      const MinibatchWords& global_words,
      const boost::shared_ptr<FactoredWeights>& global_gradient,
      const boost::shared_ptr<FactoredWeights>& adagrad);

  Real regularizerUpdate(
      const boost::shared_ptr<FactoredWeights>& global_gradient,
      Real minibatch_factor);

  void clear(const MinibatchWords& words, bool parallel_update);

  Real predict(int word_id, vector<int> context) const;

  void clearCache();

  bool operator==(const FactoredWeights& other) const;

  virtual ~FactoredWeights();

 protected:
  MatrixReal classR(int class_id) const;

  VectorReal classB(int class_id) const;

  Real getObjective(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      vector<vector<int>>& contexts,
      vector<MatrixReal>& context_vectors,
      MatrixReal& prediction_vectors,
      MatrixReal& class_probs,
      vector<VectorReal>& word_probs) const;

  virtual void getProbabilities(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const vector<vector<int>>& contexts,
      const MatrixReal& prediction_vectors,
      MatrixReal& class_probs,
      vector<VectorReal>& word_probs) const;

  MatrixReal getWeightedRepresentations(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const MatrixReal& prediction_vectors,
      const MatrixReal& class_probs,
      const vector<VectorReal>& word_probs) const;

  void getFullGradient(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const vector<vector<int>>& contexts,
      const vector<MatrixReal>& context_vectors,
      const MatrixReal& prediction_vectors,
      const MatrixReal& weighted_representations,
      MatrixReal& class_probs,
      vector<VectorReal>& word_probs,
      const boost::shared_ptr<FactoredWeights>& gradient,
      MinibatchWords& words) const;

  virtual vector<vector<int>> getNoiseWords(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices) const;

  vector<vector<int>> getNoiseClasses(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices) const;

  void estimateProjectionGradient(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const MatrixReal& prediction_vectors,
      const boost::shared_ptr<FactoredWeights>& gradient,
      MatrixReal& weighted_representations,
      Real& objective,
      MinibatchWords& words) const;

 private:
  void allocate();

  void setModelParameters();

  Block getBlock() const;

  friend class boost::serialization::access;

  template<class Archive>
  void save(Archive& ar, const unsigned int version) const {
    ar << metadata;

    ar << boost::serialization::base_object<const Weights>(*this);

    ar << index;

    ar << size;
    ar << boost::serialization::make_array(data, size);
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int version) {
    ar >> metadata;

    ar >> boost::serialization::base_object<Weights>(*this);

    ar >> index;

    ar >> size;
    data = new Real[size];
    ar >> boost::serialization::make_array(data, size);

    setModelParameters();
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER();

 protected:
  boost::shared_ptr<FactoredMetadata> metadata;
  boost::shared_ptr<WordToClassIndex> index;

  WordVectorsType S;
  WeightsType     T;
  WeightsType     FW;

  mutable ContextCache classNormalizerCache;

 private:
  int size;
  Real* data;
  vector<Mutex> mutexes;

  mutable boost::thread_specific_ptr<ClassDistribution> classDist;
  mutable boost::thread_specific_ptr<WordDistributions> wordDists;
};

} // namespace oxlm
