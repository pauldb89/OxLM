#pragma once

#include <boost/serialization/serialization.hpp>

#include "lbl/metadata.h"
#include "lbl/utils.h"

namespace oxlm {

typedef Eigen::Map<MatrixReal> ContextTransformType;
typedef vector<ContextTransformType> ContextTransformsType;
typedef Eigen::Map<MatrixReal> WordVectorsType;
typedef Eigen::Map<VectorReal> WeightsType;

class Weights {
 public:
  Weights();

  Weights(
      const ModelData& config,
      const boost::shared_ptr<Metadata>& metadata);

  Weights(
      const ModelData& config,
      const boost::shared_ptr<Metadata>& metadata,
      const boost::shared_ptr<Corpus>& training_corpus);

  boost::shared_ptr<Weights> getGradient(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      Real& objective) const;

  virtual Real getObjective(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices) const;

  bool checkGradient(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const boost::shared_ptr<Weights>& gradient);

  void update(const boost::shared_ptr<Weights>& gradient);

  void updateSquared(const boost::shared_ptr<Weights>& global_gradient);

  void updateAdaGrad(
      const boost::shared_ptr<Weights>& global_gradient,
      const boost::shared_ptr<Weights>& adagrad);

  Real regularizerUpdate(Real minibatch_factor);

  Real predict(int word_id, const vector<int>& context) const;

  void clear();

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

  boost::shared_ptr<Weights> getFullGradient(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const vector<vector<int>>& contexts,
      const vector<MatrixReal>& context_vectors,
      const MatrixReal& prediction_vectors,
      const MatrixReal& weighted_representations,
      MatrixReal& word_probs) const;

  void getContextGradient(
      const vector<int>& indices,
      const vector<vector<int>>& contexts,
      const vector<MatrixReal>& context_vectors,
      const MatrixReal& weighted_representations,
      const boost::shared_ptr<Weights>& gradient) const;

 private:
  void allocate();

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
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER();

 protected:
  ModelData config;
  boost::shared_ptr<Metadata> metadata;

  ContextTransformsType C;
  WordVectorsType       Q;
  WordVectorsType       R;
  WeightsType           B;
  WeightsType           W;

 private:
  int size;
  Real* data;
};

} // namespace oxlm
