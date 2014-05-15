#pragma once

#include "lbl/archive_export.h"
#include "lbl/feature_context_keyer.h"

namespace oxlm {

class ClassContextKeyer : public FeatureContextKeyer {
 public:
  ClassContextKeyer();

  ClassContextKeyer(int hash_space);

  virtual int getKey(const FeatureContext& feature_context) const;

  virtual NGramQuery getPrediction(
      int candidate, const FeatureContext& feature_context) const;

  bool operator==(const ClassContextKeyer& other) const;

  virtual ~ClassContextKeyer();

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & boost::serialization::base_object<FeatureContextKeyer>(*this);
    ar & hashSpace;
  }

  int hashSpace;
  hash<FeatureContext> hash_function;
};

} // namespace oxlm

BOOST_CLASS_EXPORT_KEY(oxlm::ClassContextKeyer)
