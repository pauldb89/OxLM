#pragma once

#include "lbl/factored_maxent_metadata.h"
#include "lbl/factored_weights.h"
#include "lbl/minibatch_feature_store.h"

namespace oxlm {

class MinibatchFactoredMaxentWeights : public FactoredWeights {
 public:
  MinibatchFactoredMaxentWeights(
      const boost::shared_ptr<ModelData>& config,
      const boost::shared_ptr<FactoredMaxentMetadata>& metadata);

  MinibatchFactoredMaxentWeights(
      int num_classes, const boost::shared_ptr<FactoredWeights>& base_gradient);

  void init(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& minibatch);

  void syncUpdate(
      const MinibatchWords& words,
      const boost::shared_ptr<MinibatchFactoredMaxentWeights>& gradient);

 private:
  friend class GlobalFactoredMaxentWeights;

 protected:
  boost::shared_ptr<FactoredMaxentMetadata> metadata;

  boost::shared_ptr<MinibatchFeatureStore> U;
  vector<boost::shared_ptr<MinibatchFeatureStore>> V;

 private:
  Mutex mutexU;
  vector<Mutex> mutexesV;
};

} // namespace oxlm
