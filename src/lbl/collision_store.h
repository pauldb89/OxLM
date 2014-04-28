#pragma once

#include "lbl/feature_batch.h"
#include "lbl/feature_context_keyer.h"
#include "lbl/feature_store.h"

namespace oxlm {

class CollisionStore : public virtual FeatureStore {
 public:
  CollisionStore();

  CollisionStore(const CollisionStore& other);

  CollisionStore(int vector_size, int hash_space, int feature_context_size);

  virtual VectorReal get(const vector<int>& context) const;

  virtual CollisionStore& operator=(const CollisionStore& other);

  virtual ~CollisionStore();

 protected:
  vector<int> getKeys(const vector<int>& context) const;

  boost::shared_ptr<FeatureBatch> getBatch(const pair<int, int>& batch) const;

  boost::shared_ptr<FeatureBatch> getBatch(int key, int length) const;

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void save(Archive& ar, const unsigned int version) const {
    ar << boost::serialization::base_object<FeatureStore>(*this);
    ar << vectorSize;
    ar << hashSpace;
    ar << keyer;

    ar << boost::serialization::make_array(featureWeights, hashSpace);
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int version) {
    ar >> boost::serialization::base_object<FeatureStore>(*this);
    ar >> vectorSize;
    ar >> hashSpace;
    ar >> keyer;

    featureWeights = new Real[hashSpace];
    ar >> boost::serialization::make_array(featureWeights, hashSpace);
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER();

  void deepCopy(const CollisionStore& other);

 protected:
  int vectorSize, hashSpace;
  FeatureContextKeyer keyer;
  Real* featureWeights;
};

} // namespace oxlm
