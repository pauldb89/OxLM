#pragma once

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/shared_ptr.hpp>
// Use boost::shared_ptr instead of std::shared_ptr to facilitate the
// serialization.
#include <boost/shared_ptr.hpp>

#include "lbl/archive_export.h"
#include "lbl/factored_nlm.h"
#include "lbl/global_feature_store.h"
#include "lbl/feature_store_initializer.h"
#include "lbl/unconstrained_feature_store.h"
#include "lbl/word_to_class_index.h"

using namespace std;

namespace oxlm {

class FactoredMaxentNLM : public FactoredNLM {
 public:
  FactoredMaxentNLM();

  FactoredMaxentNLM(
      const ModelData& config, const Dict& labels,
      const boost::shared_ptr<WordToClassIndex>& index,
      const FeatureStoreInitializer& initializer);

  virtual Real log_prob(
      WordId w, const vector<WordId>& context,
      bool nonlinear, bool cache) const;

  virtual void l2GradientUpdate(
      const boost::shared_ptr<MinibatchFeatureStore>& class_store,
      const vector<boost::shared_ptr<MinibatchFeatureStore>>& word_stores,
      Real minibatch_factor);

  virtual Real l2Objective(
      const boost::shared_ptr<MinibatchFeatureStore>& class_store,
      const vector<boost::shared_ptr<MinibatchFeatureStore>>& word_stores,
      Real minibatch_factor) const;

  virtual ~FactoredMaxentNLM();

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & boost::serialization::base_object<FactoredNLM>(*this);
    ar & U;
    ar & V;
  }

 public:
  boost::shared_ptr<GlobalFeatureStore> U;
  vector<boost::shared_ptr<GlobalFeatureStore>> V;
};

} // namespace oxlm

BOOST_CLASS_EXPORT_KEY(oxlm::FactoredMaxentNLM)
