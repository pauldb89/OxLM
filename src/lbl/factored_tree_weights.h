#pragma once

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include "lbl/config.h"
#include "lbl/tree_metadata.h"
#include "lbl/utils.h"
#include "lbl/weights.h"

namespace oxlm {

class FactoredTreeWeights : public Weights {
 public:
  FactoredTreeWeights();

  FactoredTreeWeights(
      const boost::shared_ptr<ModelData>& config,
      const boost::shared_ptr<TreeMetadata>& metadata);

  FactoredTreeWeights(
      const boost::shared_ptr<ModelData>& config,
      const boost::shared_ptr<TreeMetadata>& metadata,
      const boost::shared_ptr<Corpus>& training_corpus);

  FactoredTreeWeights(
      const boost::shared_ptr<ModelData>& config,
      const boost::shared_ptr<TreeMetadata>& metadata,
      const boost::shared_ptr<Corpus>& training_corpus,
      const vector<int>& indices);

  void getGradient(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const boost::shared_ptr<FactoredTreeWeights>& gradient,
      Real& objective,
      MinibatchWords& words) const;

  void estimateGradient(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const boost::shared_ptr<FactoredTreeWeights>& gradient,
      Real& objective,
      MinibatchWords& words) const;

  bool checkGradient(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const boost::shared_ptr<FactoredTreeWeights>& gradient,
      Real eps);

  Real getObjective(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices) const;

  void update(const boost::shared_ptr<FactoredTreeWeights>& gradient);

  void updateSquared(
      const MinibatchWords& global_words,
      const boost::shared_ptr<FactoredTreeWeights>& global_gradient);

  void updateAdaGrad(
      const MinibatchWords& global_words,
      const boost::shared_ptr<FactoredTreeWeights>& global_gradient,
      const boost::shared_ptr<FactoredTreeWeights>& adagrad);

  Real regularizerUpdate(
      const boost::shared_ptr<FactoredTreeWeights>& global_gradient,
      Real minibatch_factor);

  Real predict(int word_id, const vector<int>& context) const;

  void clearCache();

  MatrixReal getWordVectors() const;

  bool operator==(const FactoredTreeWeights& other) const;

 protected:
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

  const Eigen::Block<const WordVectorsType> classR(int node) const;

  Eigen::Block<WordVectorsType> classR(int node);

  const Eigen::VectorBlock<const WeightsType> classB(int node) const;

  Eigen::VectorBlock<WeightsType> classB(int node);

  vector<vector<VectorReal>> getProbabilities(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const MatrixReal& prediction_vectors) const;

  Real getObjective(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      vector<vector<int>>& contexts,
      vector<MatrixReal>& context_vectors,
      MatrixReal& prediction_vectors,
      vector<vector<VectorReal>>& probs) const;

  MatrixReal getWeightedRepresentations(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const MatrixReal& prediction_vectors,
      vector<vector<VectorReal>>& probs) const;

  boost::shared_ptr<FactoredTreeWeights> getFullGradient(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const vector<vector<int>>& contexts,
      const vector<MatrixReal>& context_vectors,
      const MatrixReal& prediction_vectors,
      const vector<vector<VectorReal>>& probs,
      const MatrixReal& weighted_representations) const;

  void getContextGradient(
      const vector<int>& indices,
      const vector<vector<int>>& contexts,
      const vector<MatrixReal>& context_vectors,
      const MatrixReal& weighted_representations,
      const boost::shared_ptr<FactoredTreeWeights>& gradient) const;

 private:
  void allocate();

  void setModelParameters();

  friend class boost::serialization::access;

  template<class Archive>
  void save(Archive& ar, const unsigned int version) const {
    ar << metadata;

    ar << config;

    ar << tree;

    ar << size;
    ar << boost::serialization::make_array(data, size);
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int version) {
    ar >> metadata;

    ar >> config;
    ar >> tree;

    ar >> size;
    data = new Real[size];
    ar >> boost::serialization::make_array(data, size);

    setModelParameters();
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER();

 protected:
  boost::shared_ptr<ModelData> config;
  boost::shared_ptr<TreeMetadata> metadata;
  boost::shared_ptr<ClassTree> tree;

  ContextTransformsType C;
  WordVectorsType       Q;
  WordVectorsType       R;
  WeightsType           B;
  WeightsType           W;

 private:
  int size;
  Real *data;
};

} // namespace oxlm
