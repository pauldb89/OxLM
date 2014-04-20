#pragma once

#include <unordered_map>
#include <unordered_set>

#include <boost/serialization/export.hpp>

#include "lbl/feature_context.h"
#include "lbl/feature_store.h"
#include "lbl/utils.h"
#include "utils/serialization_helpers.h"

using namespace std;

namespace oxlm {

class SparseFeatureStore : public FeatureStore {
 public:
  SparseFeatureStore();

  SparseFeatureStore(int vector_max_size);

  SparseFeatureStore(
      int vector_max_size, FeatureIndexesPtr feature_indexes);

  virtual VectorReal get(
      const vector<FeatureContextId>& feature_context_ids) const;

  virtual void update(
      const vector<FeatureContextId>& feature_context_ids,
      const VectorReal& values);

  virtual void l2GradientUpdate(Real sigma);

  virtual Real l2Objective(Real factor) const;

  virtual void update(const boost::shared_ptr<FeatureStore>& store);

  virtual void updateSquared(
      const boost::shared_ptr<FeatureStore>& store);

  virtual void updateAdaGrad(
      const boost::shared_ptr<FeatureStore>& gradient_store,
      const boost::shared_ptr<FeatureStore>& adagrad_store,
      Real step_size);

  virtual void clear();

  virtual size_t size() const;

  void hintFeatureIndex(
      FeatureContextId feature_context_id, int feature_index);

  bool operator==(const SparseFeatureStore& store) const;

 private:
  void update(
      const FeatureContextId& feature_context_id,
      const VectorReal& values);

  void update(
      const FeatureContextId& feature_context_id,
      const SparseVectorReal& values);

  boost::shared_ptr<SparseFeatureStore> cast(
      const boost::shared_ptr<FeatureStore>& base_store) const;

  friend class boost::serialization::access;

  template<class Archive>
  void save(Archive& ar, const unsigned int version) const {
    ar << boost::serialization::base_object<const FeatureStore>(*this);

    ar << vectorMaxSize;

    size_t num_entries = featureWeights.size();
    ar << num_entries;
    for (const auto& entry: featureWeights) {
      ar << entry.first << entry.second;
    }
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int version) {
    ar >> boost::serialization::base_object<FeatureStore>(*this);

    ar >> vectorMaxSize;

    size_t num_entries;
    ar >> num_entries;
    for (size_t i = 0; i < num_entries; ++i) {
      FeatureContextId feature_context_id;
      ar >> feature_context_id;
      SparseVectorReal weights;
      ar >> weights;
      featureWeights.insert(make_pair(feature_context_id, weights));
    }
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER();

  unordered_map<FeatureContextId, SparseVectorReal> featureWeights;
  int vectorMaxSize;
};

} // namespace oxlm

BOOST_CLASS_EXPORT_KEY(oxlm::SparseFeatureStore)
