#pragma once

#include <vector>

#include "lbl/factored_maxent_metadata.h"
#include "lbl/factored_weights.h"
#include "lbl/global_feature_store.h"
#include "lbl/minibatch_factored_maxent_weights.h"

namespace oxlm {

class GlobalFactoredMaxentWeights : public FactoredWeights {
 public:
  GlobalFactoredMaxentWeights();

  GlobalFactoredMaxentWeights(
      const boost::shared_ptr<ModelData>& config,
      const boost::shared_ptr<FactoredMaxentMetadata>& metadata);

  GlobalFactoredMaxentWeights(
      const boost::shared_ptr<ModelData>& config,
      const boost::shared_ptr<FactoredMaxentMetadata>& metadata,
      const boost::shared_ptr<Corpus>& training_corpus);

  virtual size_t numParameters() const;

  virtual void getProbabilities(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const vector<vector<int>>& contexts,
      const MatrixReal& prediction_vectors,
      MatrixReal& class_probs,
      vector<VectorReal>& word_probs) const;

  void getGradient(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const boost::shared_ptr<MinibatchFactoredMaxentWeights>& gradient,
      Real& objective,
      MinibatchWords& words) const;

  void getFullGradient(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const vector<vector<int>>& contexts,
      const vector<MatrixReal>& context_vectors,
      const MatrixReal& prediction_vectors,
      const MatrixReal& weighted_representations,
      MatrixReal& class_probs,
      vector<VectorReal>& word_probs,
      const boost::shared_ptr<MinibatchFactoredMaxentWeights>& gradient,
      MinibatchWords& words) const;

  void estimateGradient(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const boost::shared_ptr<MinibatchFactoredMaxentWeights>& gradient,
      Real& objective,
      MinibatchWords& words) const;

  bool checkGradient(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const boost::shared_ptr<MinibatchFactoredMaxentWeights>& gradient,
      Real eps);

  void updateSquared(
      const MinibatchWords& global_words,
      const boost::shared_ptr<MinibatchFactoredMaxentWeights>& global_gradient);

  void updateAdaGrad(
      const MinibatchWords& global_words,
      const boost::shared_ptr<MinibatchFactoredMaxentWeights>& global_gradient,
      const boost::shared_ptr<GlobalFactoredMaxentWeights>& adagrad);

  Real regularizerUpdate(
      const boost::shared_ptr<MinibatchFactoredMaxentWeights>& global_gradient,
      Real minibatch_factor);

  Real predict(int word_id, vector<int> context) const;

  bool operator==(const GlobalFactoredMaxentWeights& other) const;

 protected:
  bool checkGradientStore(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      const boost::shared_ptr<GlobalFeatureStore>& store,
      const boost::shared_ptr<MinibatchFeatureStore>& gradient_store,
      Real eps);

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & metadata;

    ar & boost::serialization::base_object<FactoredWeights>(*this);

    ar & U;
    ar & V;
  }

  void initialize();

 protected:
  boost::shared_ptr<FactoredMaxentMetadata> metadata;

  boost::shared_ptr<GlobalFeatureStore> U;
  vector<boost::shared_ptr<GlobalFeatureStore>> V;
};

} // namespace oxlm
