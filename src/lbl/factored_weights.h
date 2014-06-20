#pragma once

#include "lbl/factored_metadata.h"
#include "lbl/weights.h"
#include "lbl/word_to_class_index.h"

namespace oxlm {

class FactoredWeights : public Weights {
 public:
  FactoredWeights(
      const ModelData& config,
      const boost::shared_ptr<FactoredMetadata>& metadata);

  FactoredWeights(
      const ModelData& config,
      const boost::shared_ptr<FactoredMetadata>& metadata,
      const boost::shared_ptr<Corpus>& training_corpus);

  boost::shared_ptr<FactoredWeights> getGradient(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      Real& objective) const;

  virtual Real getObjective(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices) const;

  bool checkGradient(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const boost::shared_ptr<FactoredWeights>& gradient);

  void update(const boost::shared_ptr<FactoredWeights>& gradient);

  void updateSquared(const boost::shared_ptr<FactoredWeights>& global_gradient);

  void updateAdaGrad(
      const boost::shared_ptr<FactoredWeights>& global_gradient,
      const boost::shared_ptr<FactoredWeights>& adagrad);

  Real regularizerUpdate(Real minibatch_factor);

  void clear();

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

  void getProbabilities(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const MatrixReal& prediction_vectors,
      MatrixReal& class_probs,
      vector<VectorReal>& word_probs) const;

  MatrixReal getWeightedRepresentations(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const MatrixReal& prediction_vectors,
      const MatrixReal& class_probs,
      const vector<VectorReal>& word_probs) const;

  boost::shared_ptr<FactoredWeights> getFullGradient(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const vector<vector<int>>& contexts,
      const vector<MatrixReal>& context_vectors,
      const MatrixReal& prediction_vectors,
      const MatrixReal& weighted_representations,
      MatrixReal& class_probs,
      vector<VectorReal>& word_probs) const;

 private:
  void allocate();

 protected:
  boost::shared_ptr<FactoredMetadata> metadata;
  boost::shared_ptr<WordToClassIndex> index;

  WordVectorsType S;
  WeightsType     T;
  WeightsType     FW;

 private:
  int size;
  Real* data;
};

} // namespace oxlm
