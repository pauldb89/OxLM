#pragma once

#include <unordered_map>
#include <unordered_set>

#include "lbl/feature_context_extractor.h"
#include "lbl/minibatch_feature_store.h"
#include "lbl/utils.h"
#include "utils/serialization_helpers.h"

using namespace std;

namespace oxlm {

class SparseMinibatchFeatureStore : public MinibatchFeatureStore {
 public:
  SparseMinibatchFeatureStore();

  SparseMinibatchFeatureStore(
      int vector_max_size,
      const boost::shared_ptr<FeatureContextExtractor>& extractor);

  SparseMinibatchFeatureStore(
      int vector_max_size,
      MinibatchFeatureIndexesPtr feature_indexes,
      const boost::shared_ptr<FeatureContextExtractor>& extractor);

  virtual VectorReal get(const vector<int>& context) const;

  virtual void update(const vector<int>& context, const VectorReal& values);

  virtual void update(const boost::shared_ptr<MinibatchFeatureStore>& store);

  virtual void clear();

  virtual Real getFeature(const pair<int, int>& index) const;

  virtual size_t size() const;

  void hintFeatureIndex(int feature_context_id, int feature_index);

  static boost::shared_ptr<SparseMinibatchFeatureStore> cast(
      const boost::shared_ptr<MinibatchFeatureStore>& base_store);

  bool operator==(const SparseMinibatchFeatureStore& store) const;

  virtual ~SparseMinibatchFeatureStore();

 private:
  void update(int feature_context_id, const VectorReal& values);

  void update(int feature_context_id, const SparseVectorReal& values);

  friend class SparseGlobalFeatureStore;

  int vectorMaxSize;
  boost::shared_ptr<FeatureContextExtractor> extractor;
  unordered_map<int, SparseVectorReal> featureWeights;
};

} // namespace oxlm
