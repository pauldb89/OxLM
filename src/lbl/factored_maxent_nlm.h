#pragma once

#include "lbl/factored_nlm.h"
#include "lbl/unconstrained_feature_store.h"

namespace oxlm {

class FactoredMaxentNLM : public FactoredNLM {
 public:
  FactoredMaxentNLM();

  FactoredMaxentNLM(const ModelData& config, const Dict& labels,
                    const vector<int>& classes);

  virtual Real log_prob(
      WordId w, const vector<WordId>& context,
      bool nonlinear, bool cache) const;

  virtual Real l2_gradient_update(Real lambda);

  friend class boost::serialization::access;

  template<class Archive>
  void save(Archive& ar, const unsigned int version) const {
    FactoredNLM::save(ar, version);
    ar << U << V;
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int version) {
    FactoredNLM::load(ar, version);
    ar >> U >> V;
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER();

  UnconstrainedFeatureStore U;
  vector<UnconstrainedFeatureStore> V;
};

} // namespace oxlm
