#pragma once

#include <unordered_map>
#include <unordered_set>

#include <boost/serialization/shared_ptr.hpp>

#include "lbl/archive_export.h"
#include "lbl/feature_context_extractor.h"
#include "lbl/global_feature_store.h"
#include "lbl/utils.h"
#include "utils/serialization_helpers.h"

using namespace std;

namespace oxlm {

class SparseGlobalFeatureStore : public GlobalFeatureStore {
 public:
  SparseGlobalFeatureStore();

  SparseGlobalFeatureStore(
      int vector_max_size, int num_contexts,
      const boost::shared_ptr<FeatureContextExtractor>& extractor);

  SparseGlobalFeatureStore(
      int vector_max_size, GlobalFeatureIndexesPtr feature_indexes,
      const boost::shared_ptr<FeatureContextExtractor>& extractor);

  virtual VectorReal get(const vector<int>& context) const;

  virtual void l2GradientUpdate(
      const boost::shared_ptr<MinibatchFeatureStore>& base_minibatch_store,
      Real sigma);

  virtual Real l2Objective(
      const boost::shared_ptr<MinibatchFeatureStore>& base_minibatch_store,
      Real factor) const;

  virtual void updateSquared(
      const boost::shared_ptr<MinibatchFeatureStore>& base_minibatch_store);

  virtual void updateAdaGrad(
      const boost::shared_ptr<MinibatchFeatureStore>& gradient_store,
      const boost::shared_ptr<GlobalFeatureStore>& adagrad_store,
      Real step_size);

  virtual size_t size() const;

  void hintFeatureIndex(int feature_context_id, int feature_index);

  static boost::shared_ptr<SparseGlobalFeatureStore> cast(
      const boost::shared_ptr<GlobalFeatureStore>& base_store);

  bool operator==(const SparseGlobalFeatureStore& store) const;

  bool operator==(const boost::shared_ptr<GlobalFeatureStore>& other) const;

  virtual vector<pair<int, int>> getFeatureIndexes() const;

  void updateFeature(const pair<int, int>& index, Real value);

  virtual ~SparseGlobalFeatureStore();

 private:
  void update(int feature_context_id, const SparseVectorReal& values);

  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & boost::serialization::base_object<GlobalFeatureStore>(*this);

    ar & vectorMaxSize;
    ar & extractor;
    ar & featureWeights;
  }

  friend class SparseMinibatchFeatureStore;

  int vectorMaxSize;
  boost::shared_ptr<FeatureContextExtractor> extractor;
  vector<SparseVectorReal> featureWeights;
};

} // namespace oxlm

BOOST_CLASS_EXPORT_KEY(oxlm::SparseGlobalFeatureStore)
