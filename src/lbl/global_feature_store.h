#pragma once

#include "lbl/feature_store.h"
#include "lbl/minibatch_feature_store.h"

namespace oxlm {

class GlobalFeatureStore : virtual public FeatureStore {
 public:
  virtual void l2GradientUpdate(
      const boost::shared_ptr<MinibatchFeatureStore>& base_minibatch_store,
      Real sigma) = 0;

  virtual Real l2Objective(
      const boost::shared_ptr<MinibatchFeatureStore>& base_minibatch_store,
      Real sigma) const = 0;

  virtual void updateSquared(
      const boost::shared_ptr<MinibatchFeatureStore>& base_minibatch_store) = 0;

  virtual void updateAdaGrad(
      const boost::shared_ptr<MinibatchFeatureStore>& base_gradient_store,
      const boost::shared_ptr<GlobalFeatureStore>& base_adagrad_store,
      Real step_size) = 0;

  virtual vector<pair<int, int>> getFeatureIndexes() const = 0;

  virtual void updateFeature(const pair<int, int>& index, Real value) = 0;

  virtual ~GlobalFeatureStore();

  virtual bool operator==(const boost::shared_ptr<GlobalFeatureStore>& other) const = 0;

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & boost::serialization::base_object<FeatureStore>(*this);
  }
};

} // namespace oxlm
