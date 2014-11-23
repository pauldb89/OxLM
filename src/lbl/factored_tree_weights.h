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

  virtual void printInfo() const;

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

  Real getLogLikelihood(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices) const;

  virtual Real getLogProb(int word_id, vector<int> context) const;

  virtual Real getUnnormalizedScore(int word, const vector<int>& context) const;

  MatrixReal getWordVectors() const;

  bool operator==(const FactoredTreeWeights& other) const;

  ~FactoredTreeWeights();

 protected:
  const Eigen::Block<const WordVectorsType> classR(int node) const;

  Eigen::Block<WordVectorsType> classR(int node);

  const Eigen::VectorBlock<const WeightsType> classB(int node) const;

  Eigen::VectorBlock<WeightsType> classB(int node);

  vector<vector<VectorReal>> getProbabilities(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const vector<MatrixReal>& forward_weights) const;

  Real getObjective(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      vector<vector<int>>& contexts,
      vector<MatrixReal>& context_vectors,
      vector<MatrixReal>& forward_weights,
      vector<vector<VectorReal>>& probs) const;

  void getFullGradient(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const vector<vector<int>>& contexts,
      const vector<MatrixReal>& context_vectors,
      const vector<MatrixReal>& forward_weights,
      vector<vector<VectorReal>>& probs,
      const boost::shared_ptr<FactoredTreeWeights>& gradient,
      MinibatchWords& words) const;

  void getProjectionGradient(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const vector<MatrixReal>& forward_weights,
      vector<vector<VectorReal>>& probs,
      const boost::shared_ptr<FactoredTreeWeights>& gradient,
      MatrixReal& backward_weights,
      MinibatchWords& words) const;

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
  boost::shared_ptr<TreeMetadata> metadata;
  boost::shared_ptr<ClassTree> tree;
};

} // namespace oxlm
