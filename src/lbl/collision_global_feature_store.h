#pragma once

#include "lbl/archive_export.h"
#include "lbl/collision_minibatch_feature_store.h"
#include "lbl/feature_context_keyer.h"
#include "lbl/global_feature_store.h"

namespace oxlm {

class CollisionGlobalFeatureStore : public GlobalFeatureStore {
 public:
  CollisionGlobalFeatureStore();

  CollisionGlobalFeatureStore(const CollisionGlobalFeatureStore& other);

  CollisionGlobalFeatureStore(
      int vector_size, int hash_space, int feature_context_size);

  virtual VectorReal get(const vector<int>& context) const;

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

  virtual CollisionGlobalFeatureStore& operator=(
      const CollisionGlobalFeatureStore& other);

  virtual ~CollisionGlobalFeatureStore();

 private:
  void deepCopy(const CollisionGlobalFeatureStore& other);

  friend class boost::serialization::access;

  template<class Archive>
  void save(Archive& ar, const unsigned int version) const {
    ar & boost::serialization::base_object<GlobalFeatureStore>(*this);

    ar << boost::serialization::base_object<FeatureStore>(*this);
    ar << vectorSize;
    ar << hashSpace;
    ar << keyer;

    ar << boost::serialization::make_array(featureWeights, hashSpace);
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int version) {
    ar >> boost::serialization::base_object<GlobalFeatureStore>(*this);

    ar >> vectorSize;
    ar >> hashSpace;
    ar >> keyer;

    featureWeights = new Real[hashSpace];
    ar >> boost::serialization::make_array(featureWeights, hashSpace);
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER();

  int vectorSize, hashSpace;
  FeatureContextKeyer keyer;
  Real* featureWeights;
};

} // namespace oxlm

BOOST_CLASS_EXPORT_KEY(oxlm::CollisionGlobalFeatureStore)
