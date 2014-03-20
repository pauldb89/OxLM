#pragma once

#include <unordered_map>

#include "lbl/feature_context.h"
#include "lbl/feature_store.h"
#include "lbl/utils.h"

using namespace std;

namespace oxlm {

class UnconstrainedFeatureStore : public FeatureStore {
 public:
  UnconstrainedFeatureStore();

  UnconstrainedFeatureStore(int vector_size);

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

  bool operator==(const UnconstrainedFeatureStore& store) const;

 private:
  void update(
      const FeatureContextId& feature_context_id,
      const VectorReal& values);

  boost::shared_ptr<UnconstrainedFeatureStore> cast(
      const boost::shared_ptr<FeatureStore>& base_store) const;

  friend class boost::serialization::access;

  template<class Archive>
  void save(Archive& ar, const unsigned int version) const {
    ar << boost::serialization::base_object<const FeatureStore>(*this);

    ar << vectorSize;

    size_t num_entries = featureWeights.size();
    ar << num_entries;
    for (const auto& entry: featureWeights) {
      ar << entry.first;
      const VectorReal weights = entry.second;
      ar << boost::serialization::make_array(weights.data(), weights.rows());
    }
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int version) {
    ar >> boost::serialization::base_object<FeatureStore>(*this);

    ar >> vectorSize;

    size_t num_entries;
    ar >> num_entries;
    for (size_t i = 0; i < num_entries; ++i) {
      FeatureContextId feature_context_id;
      ar >> feature_context_id;

      VectorReal weights = VectorReal::Zero(vectorSize);
      ar >> boost::serialization::make_array(weights.data(), vectorSize);

      featureWeights.insert(make_pair(feature_context_id, weights));
    }
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER();

  unordered_map<FeatureContextId, VectorReal> featureWeights;
  int vectorSize;

};

} // namespace oxlm
