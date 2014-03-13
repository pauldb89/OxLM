#pragma once

#include <unordered_map>

#include "lbl/feature_context.h"
#include "lbl/feature_store.h"
#include "lbl/utils.h"

using namespace std;

namespace oxlm {

class SparseFeatureStore : public FeatureStore {
 public:
  SparseFeatureStore();

  SparseFeatureStore(int vector_max_size);

  SparseFeatureStore(
      int vector_max_size, const MatchingContexts& matching_contexts);

  virtual VectorReal get(
      const vector<FeatureContext>& feature_contexts) const;

  virtual void update(
      const vector<FeatureContext>& feature_contexts,
      const VectorReal& values);

  virtual Real updateRegularizer(Real lambda);

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
      const vector<FeatureContext>& feature_context,
      int feature_index);

  bool operator==(const SparseFeatureStore& store) const;

 private:
  void update(
      const FeatureContext& feature_context,
      const VectorReal& values);

  void update(
      const FeatureContext& feature_context,
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
      FeatureContext feature_context;
      ar >> feature_context;
      SparseVectorReal weights;
      ar >> weights;
      featureWeights.insert(make_pair(feature_context, weights));
    }
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER();

  unordered_map<FeatureContext, SparseVectorReal, hash<FeatureContext>>
      featureWeights;
  int vectorMaxSize;
};

template<class Scalar>
struct CwiseIdentityOp {
  const Scalar operator()(const Scalar& x) const {
    return 1.0;
  }
};

template<class Scalar>
struct CwiseNumeratorOp {
  CwiseNumeratorOp(Scalar eps) : eps(eps) {}

  const Scalar operator()(const Scalar& x) const {
    return fabs(x) < eps ? 1.0 : 1.0 / x;
  }

  Scalar eps;
};

} // namespace oxlm

namespace boost {
namespace serialization {

template<class Archive, class Scalar>
inline void save(
    Archive& ar, const Eigen::SparseVector<Scalar>& v,
    const unsigned int version) {
  int max_size = v.size();
  ar << max_size;
  int actual_size = v.nonZeros();
  ar << actual_size;
  ar << boost::serialization::make_array(&v._data().index(0), actual_size);
  ar << boost::serialization::make_array(&v._data().value(0), actual_size);
}

template<class Archive, class Scalar>
inline void load(
    Archive& ar, Eigen::SparseVector<Scalar>& v, const unsigned int version) {
  int max_size;
  ar >> max_size;
  v.resize(max_size);
  int actual_size;
  ar >> actual_size;
  v.resizeNonZeros(actual_size);
  ar >> boost::serialization::make_array(&v._data().index(0), actual_size);
  ar >> boost::serialization::make_array(&v._data().value(0), actual_size);
}

template<class Archive, class Scalar>
inline void serialize(
    Archive& ar, Eigen::SparseVector<Scalar>& v, const unsigned int version) {
  boost::serialization::split_free(ar, v, version);
}

} // namespace serialization
} // namespace boost


