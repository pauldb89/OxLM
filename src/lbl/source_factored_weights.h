#pragma once

#include "lbl/factored_weights.h"

namespace oxlm {

class SourceFactoredWeights : public FactoredWeights {
 public:
  SourceFactoredWeights();

  SourceFactoredWeights(
      const boost::shared_ptr<ModelData>& config,
      const boost::shared_ptr<FactoredMetadata>& metadata);

  SourceFactoredWeights(
      const boost::shared_ptr<ModelData>& config,
      const boost::shared_ptr<FactoredMetadata>& metadata,
      const boost::shared_ptr<Corpus>& training_corpus);

  SourceFactoredWeights(
      const SourceFactoredWeights& weights);

  bool checkGradient(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const boost::shared_ptr<SourceFactoredWeights>& gradient,
      double eps);

  virtual size_t numParameters() const;

  virtual void printInfo() const;

  void init(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& minibatch);

  void syncUpdate(
      const MinibatchWords& words,
      const boost::shared_ptr<SourceFactoredWeights>& gradient);

  void updateSquared(
      const MinibatchWords& global_words,
      const boost::shared_ptr<SourceFactoredWeights>& global_gradient);

  void updateAdaGrad(
      const MinibatchWords& global_words,
      const boost::shared_ptr<SourceFactoredWeights>& global_gradient,
      const boost::shared_ptr<SourceFactoredWeights>& adagrad);

  Real regularizerUpdate(
      const boost::shared_ptr<SourceFactoredWeights>& global_gradient,
      Real minibatch_factor);

  void clear(const MinibatchWords& words, bool parallel_update);

  ~SourceFactoredWeights();

 protected:
  virtual void getContextVectors(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      vector<vector<int>>& contexts,
      vector<MatrixReal>& context_vectors) const;

  virtual void setContextWords(
      const vector<vector<int>>& contexts,
      MinibatchWords& words) const;

  virtual MatrixReal getPredictionVectors(
      const vector<int>& indices,
      const vector<MatrixReal>& context_vectors) const;

  virtual void getContextGradient(
      const vector<int>& indices,
      const vector<vector<int>>& contexts,
      const vector<MatrixReal>& context_vectors,
      const MatrixReal& weighted_representations,
      const boost::shared_ptr<Weights>& base_gradient) const;

  MatrixReal getSourceContextProduct(
      int index, const MatrixReal& weighted_representations,
      bool transpose = false) const;

 private:
  void allocate();

  void setModelParameters();

  Block getBlock(int start, int size) const;

 protected:
  ContextTransformsType SC;
  WordVectorsType SQ;
  WeightsType SW;

 private:
  Real* data;
  int size;
  vector<Mutex> mutexesSQ;
  vector<Mutex> mutexesSC;
};

} // namespace oxlm
