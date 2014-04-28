#pragma once

#include "lbl/archive_export.h"
#include "lbl/collision_minibatch_feature_store.h"
#include "lbl/collision_store.h"
#include "lbl/global_feature_store.h"

namespace oxlm {

class CollisionGlobalFeatureStore
    : public virtual CollisionStore, public GlobalFeatureStore {
 public:
  CollisionGlobalFeatureStore();

  CollisionGlobalFeatureStore(
      int vector_size, int hash_space, int feature_context_size);

  virtual void l2GradientUpdate(
      const boost::shared_ptr<MinibatchFeatureStore>& store, Real sigma);

  virtual Real l2Objective(
      const boost::shared_ptr<MinibatchFeatureStore>& store, Real sigma) const;

  virtual void updateSquared(
      const boost::shared_ptr<MinibatchFeatureStore>& store);

  virtual void updateAdaGrad(
      const boost::shared_ptr<MinibatchFeatureStore>& gradient_store,
      const boost::shared_ptr<GlobalFeatureStore>& adagrad_store,
      Real step_size);

  virtual size_t size() const;

  static boost::shared_ptr<CollisionGlobalFeatureStore> cast(
      const boost::shared_ptr<GlobalFeatureStore>& base_store);

  bool operator==(const CollisionGlobalFeatureStore& other) const;

  virtual ~CollisionGlobalFeatureStore();

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & boost::serialization::base_object<CollisionStore>(*this);
    ar & boost::serialization::base_object<GlobalFeatureStore>(*this);
  }
};

} // namespace oxlm

BOOST_CLASS_EXPORT_KEY(oxlm::CollisionGlobalFeatureStore)
