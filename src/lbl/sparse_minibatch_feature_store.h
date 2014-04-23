#pragma once

#include <unordered_map>
#include <unordered_set>

#include "lbl/feature_context.h"
#include "lbl/minibatch_feature_store.h"
#include "lbl/utils.h"
#include "utils/serialization_helpers.h"

using namespace std;

namespace oxlm {

class SparseMinibatchFeatureStore : public MinibatchFeatureStore {
 public:
  SparseMinibatchFeatureStore();

  SparseMinibatchFeatureStore(int vector_max_size);

  SparseMinibatchFeatureStore(
      int vector_max_size, MinibatchFeatureIndexesPtr feature_indexes);

  virtual VectorReal get(const vector<int>& feature_context_ids) const;

  virtual void update(
      const vector<int>& feature_context_ids,
      const VectorReal& values);

  virtual void update(const boost::shared_ptr<MinibatchFeatureStore>& store);

  virtual void clear();

  virtual size_t size() const;

  void hintFeatureIndex(int feature_context_id, int feature_index);

  static boost::shared_ptr<SparseMinibatchFeatureStore> cast(
      const boost::shared_ptr<MinibatchFeatureStore>& base_store);

  bool operator==(const SparseMinibatchFeatureStore& store) const;

 private:
  void update(int feature_context_id, const VectorReal& values);

  void update(int feature_context_id, const SparseVectorReal& values);

  friend class SparseGlobalFeatureStore;

  unordered_map<int, SparseVectorReal> featureWeights;
  int vectorMaxSize;
};

} // namespace oxlm
