#pragma once

#include <unordered_map>

// We need to include the archives in the shared library so BOOST_EXPORT knows
// to register implementations for all archive/derived class pairs.
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/export.hpp>

#include "lbl/feature_context.h"
#include "lbl/global_feature_store.h"
#include "lbl/minibatch_feature_store.h"
#include "lbl/utils.h"

using namespace std;

namespace oxlm {

class UnconstrainedFeatureStore :
    public GlobalFeatureStore, public MinibatchFeatureStore {
 public:
  UnconstrainedFeatureStore();

  UnconstrainedFeatureStore(int vector_size);

  virtual VectorReal get(const vector<int>& feature_context_ids) const;

  virtual void update(
      const vector<int>& feature_context_ids,
      const VectorReal& values);

  virtual void l2GradientUpdate(
      const boost::shared_ptr<MinibatchFeatureStore>& base_minibatch_store,
      Real sigma);

  virtual Real l2Objective(
      const boost::shared_ptr<MinibatchFeatureStore>& base_minibatch_store,
      Real factor) const;

  virtual void update(const boost::shared_ptr<MinibatchFeatureStore>& store);

  virtual void updateSquared(
      const boost::shared_ptr<MinibatchFeatureStore>& store);

  virtual void updateAdaGrad(
      const boost::shared_ptr<MinibatchFeatureStore>& gradient_store,
      const boost::shared_ptr<GlobalFeatureStore>& adagrad_store,
      Real step_size);

  virtual void clear();

  virtual size_t size() const;

  bool operator==(const UnconstrainedFeatureStore& store) const;

 private:
  void update(int feature_context_id, const VectorReal& values);

  boost::shared_ptr<UnconstrainedFeatureStore> cast(
      const boost::shared_ptr<FeatureStore>& base_store) const;

  friend class boost::serialization::access;

  template<class Archive>
  void save(Archive& ar, const unsigned int version) const {
    ar << boost::serialization::base_object<const GlobalFeatureStore>(*this);
    ar << boost::serialization::base_object<const MinibatchFeatureStore>(*this);

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
    ar >> boost::serialization::base_object<GlobalFeatureStore>(*this);
    ar >> boost::serialization::base_object<MinibatchFeatureStore>(*this);

    ar >> vectorSize;

    size_t num_entries;
    ar >> num_entries;
    for (size_t i = 0; i < num_entries; ++i) {
      int feature_context_id;
      ar >> feature_context_id;

      VectorReal weights = VectorReal::Zero(vectorSize);
      ar >> boost::serialization::make_array(weights.data(), vectorSize);

      featureWeights.insert(make_pair(feature_context_id, weights));
    }
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER();

  unordered_map<int, VectorReal> featureWeights;
  int vectorSize;

};

} // namespace oxlm

BOOST_CLASS_EXPORT_KEY(oxlm::UnconstrainedFeatureStore)
