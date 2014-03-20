#pragma once

#include <vector>

#include <boost/shared_ptr.hpp>
#include <boost/serialization/serialization.hpp>

#include "lbl/utils.h"
#include "lbl/feature_context.h"

using namespace std;

namespace oxlm {

class FeatureStore {
 public:
  virtual VectorReal get(
      const vector<FeatureContextId>& feature_context_ids) const = 0;

  virtual void update(
      const vector<FeatureContextId>& feature_context_ids,
      const VectorReal& values) = 0;

  virtual void l2GradientUpdate(Real lambda) = 0;

  virtual Real l2Objective(Real factor) const = 0;

  virtual void update(const boost::shared_ptr<FeatureStore>& store) = 0;

  virtual void updateSquared(
      const boost::shared_ptr<FeatureStore>& store) = 0;

  virtual void updateAdaGrad(
      const boost::shared_ptr<FeatureStore>& gradient_store,
      const boost::shared_ptr<FeatureStore>& adagrad_store,
      Real step_size) = 0;

  virtual void clear() = 0;

  virtual size_t size() const = 0;

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void save(Archive& ar, const unsigned int version) const {}

  template<class Archive>
  void load(Archive& ar, const unsigned int version) {}

  BOOST_SERIALIZATION_SPLIT_MEMBER();
};

} // namespace oxlm
