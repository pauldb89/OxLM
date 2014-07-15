#pragma once

#include <unordered_map>

#include <boost/serialization/shared_ptr.hpp>

#include "lbl/archive_export.h"
#include "lbl/feature_context_extractor.h"
#include "lbl/global_feature_store.h"
#include "lbl/minibatch_feature_store.h"
#include "lbl/utils.h"

using namespace std;

namespace oxlm {

class UnconstrainedFeatureStore :
    public GlobalFeatureStore, public MinibatchFeatureStore {
 public:
  UnconstrainedFeatureStore();

  UnconstrainedFeatureStore(
      int vector_size,
      const boost::shared_ptr<FeatureContextExtractor>& extractor);

  virtual VectorReal get(const vector<int>& context) const;

  virtual void update(const vector<int>& context, const VectorReal& values);

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

  static boost::shared_ptr<UnconstrainedFeatureStore> cast(
      const boost::shared_ptr<FeatureStore>& base_store);

  bool operator==(const UnconstrainedFeatureStore& store) const;

  bool operator==(const boost::shared_ptr<GlobalFeatureStore>& other) const;

  virtual vector<pair<int, int>> getFeatureIndexes() const;

  virtual void updateFeature(const pair<int, int>& index, Real value);

  virtual Real getFeature(const pair<int, int>& index) const;

  virtual ~UnconstrainedFeatureStore();

 private:
  void update(int feature_context_id, const VectorReal& values);

  friend class boost::serialization::access;

  template<class Archive>
  void save(Archive& ar, const unsigned int version) const {
    ar << boost::serialization::base_object<const GlobalFeatureStore>(*this);
    ar << boost::serialization::base_object<const MinibatchFeatureStore>(*this);

    ar << vectorSize;
    ar << extractor;

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
    ar >> extractor;

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

  int vectorSize;
  boost::shared_ptr<FeatureContextExtractor> extractor;
  unordered_map<int, VectorReal> featureWeights;
};

} // namespace oxlm

BOOST_CLASS_EXPORT_KEY(oxlm::UnconstrainedFeatureStore)
