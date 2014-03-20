#pragma once

// Use boost::shared_ptr instead of std::shared_ptr to facilitate serialization.
#include <boost/shared_ptr.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include "lbl/feature_context_extractor.h"
#include "lbl/factored_nlm.h"
#include "lbl/feature_store.h"
#include "lbl/feature_store_initializer.h"
#include "lbl/sparse_feature_store.h"
#include "lbl/unconstrained_feature_store.h"
#include "lbl/word_to_class_index.h"

using namespace std;

namespace oxlm {

class FactoredMaxentNLM : public FactoredNLM {
 public:
  FactoredMaxentNLM(
      const ModelData& config, const Dict& labels,
      const WordToClassIndex& index,
      const boost::shared_ptr<FeatureContextExtractor>& extractor,
      const FeatureStoreInitializer& initializer);

  virtual Real log_prob(
      WordId w, const vector<WordId>& context,
      bool nonlinear, bool cache) const;

  virtual void l2GradientUpdate(Real minibatch_factor);

  virtual Real l2Objective(Real minibatch_factor) const;

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void save(Archive& ar, const unsigned int version) const {
    ar << boost::serialization::base_object<const FactoredNLM>(*this);
    ar << *extractor << U << V;
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int version) {
    ar >> boost::serialization::base_object<FactoredNLM>(*this);
    ar.register_type(static_cast<UnconstrainedFeatureStore*>(NULL));
    ar.register_type(static_cast<SparseFeatureStore*>(NULL));
    ar >> *extractor >> U >> V;
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER();

 public:
  boost::shared_ptr<FeatureContextExtractor> extractor;
  boost::shared_ptr<FeatureStore> U;
  vector<boost::shared_ptr<FeatureStore>> V;
};

} // namespace oxlm
