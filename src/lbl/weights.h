#pragma once

#include <mutex>
#include <vector>

#include <boost/make_shared.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/shared_ptr.hpp>
#include <boost/thread/tss.hpp>

#include "lbl/context_cache.h"
#include "lbl/metadata.h"
#include "lbl/minibatch_words.h"
#include "lbl/utils.h"

namespace oxlm {

typedef Eigen::Map<MatrixReal> ContextTransformType;
typedef vector<ContextTransformType> ContextTransformsType;
typedef Eigen::Map<MatrixReal> WordVectorsType;
typedef Eigen::Map<VectorReal> WeightsType;

typedef boost::shared_ptr<mutex> Mutex;
typedef pair<size_t, size_t> Block;

class Weights {
 public:
  Weights();

  Weights(
      const boost::shared_ptr<ModelData>& config,
      const boost::shared_ptr<Metadata>& metadata);

  Weights(
      const boost::shared_ptr<ModelData>& config,
      const boost::shared_ptr<Metadata>& metadata,
      const boost::shared_ptr<Corpus>& training_corpus);

  Weights(const Weights& other);

  virtual size_t numParameters() const;

  void init(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& minibatch);

  void getGradient(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const boost::shared_ptr<Weights>& gradient,
      Real& objective,
      MinibatchWords& words) const;

  virtual Real getObjective(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices) const;

  bool checkGradient(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const boost::shared_ptr<Weights>& gradient,
      double eps);

  void estimateGradient(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const boost::shared_ptr<Weights>& gradient,
      Real& objective,
      MinibatchWords& words) const;

  void syncUpdate(
      const MinibatchWords& words,
      const boost::shared_ptr<Weights>& gradient);

  void updateSquared(
      const MinibatchWords& global_words,
      const boost::shared_ptr<Weights>& global_gradient);

  void updateAdaGrad(
      const MinibatchWords& global_words,
      const boost::shared_ptr<Weights>& global_gradient,
      const boost::shared_ptr<Weights>& adagrad);

  Real regularizerUpdate(
      const boost::shared_ptr<Weights>& global_gradient,
      Real minibatch_factor);

  void clear(const MinibatchWords& words, bool parallel_update);

  Real predict(int word_id, vector<int> context) const;

  void clearCache();

  MatrixReal getWordVectors() const;

  bool operator==(const Weights& other) const;

  virtual ~Weights();

 protected:

  Real getObjective(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      vector<vector<int>>& contexts,
      vector<MatrixReal>& context_vectors,
      MatrixReal& prediction_vectors,
      MatrixReal& word_probs) const;

  void getContextVectors(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      vector<vector<int>>& contexts,
      vector<MatrixReal>& context_vectors) const;

  void setContextWords(
      const vector<vector<int>>& contexts,
      MinibatchWords& words) const;

  MatrixReal getPredictionVectors(
      const vector<int>& indices,
      const vector<MatrixReal>& context_vectors) const;

  MatrixReal getContextProduct(
      int index, const MatrixReal& representations,
      bool transpose = false) const;

  MatrixReal getProbabilities(
      const vector<int>& indices,
      const MatrixReal& prediction_vectors) const;

  MatrixReal getWeightedRepresentations(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const MatrixReal& prediction_vectors,
      const MatrixReal& word_probs) const;

  void getFullGradient(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const vector<vector<int>>& contexts,
      const vector<MatrixReal>& context_vectors,
      const MatrixReal& prediction_vectors,
      const MatrixReal& weighted_representations,
      MatrixReal& word_probs,
      const boost::shared_ptr<Weights>& gradient,
      MinibatchWords& words) const;

  void getContextGradient(
      const vector<int>& indices,
      const vector<vector<int>>& contexts,
      const vector<MatrixReal>& context_vectors,
      const MatrixReal& weighted_representations,
      const boost::shared_ptr<Weights>& gradient) const;

  virtual vector<vector<int>> getNoiseWords(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices) const;

  void estimateProjectionGradient(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const MatrixReal& prediction_vectors,
      const boost::shared_ptr<Weights>& gradient,
      MatrixReal& weighted_representations,
      Real& objective,
      MinibatchWords& words) const;

  VectorReal getPredictionVector(const vector<int>& context) const;

 private:
  void allocate();

  void setModelParameters();

  Block getBlock(int start, int size) const;

  friend class boost::serialization::access;

  template<class Archive>
  void save(Archive& ar, const unsigned int version) const {
    ar << config;
    ar << metadata;

    ar << size;
    ar << boost::serialization::make_array(data, size);
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int version) {
    ar >> config;

    ar >> metadata;

    ar >> size;
    data = new Real[size];
    ar >> boost::serialization::make_array(data, size);

    setModelParameters();
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER();

 protected:
  boost::shared_ptr<ModelData> config;
  boost::shared_ptr<Metadata> metadata;

  ContextTransformsType C;
  WordVectorsType       Q;
  WordVectorsType       R;
  WeightsType           B;
  WeightsType           W;

  mutable ContextCache normalizerCache;

 private:
  int size;
  Real* data;
  vector<Mutex> mutexesC;
  vector<Mutex> mutexesQ;
  vector<Mutex> mutexesR;
  Mutex mutexB;
};

} // namespace oxlm
