#pragma once

#include <boost/serialization/shared_ptr.hpp>

#include "lbl/archive_export.h"
#include "lbl/feature_context_extractor.h"
#include "lbl/feature_context_hasher.h"

namespace oxlm {

class WordContextExtractor : public FeatureContextExtractor {
 public:
  WordContextExtractor();

  WordContextExtractor(
      int class_id, const boost::shared_ptr<FeatureContextHasher>& hasher);

  virtual vector<int> getFeatureContextIds(const vector<int>& context) const;

  bool operator==(const WordContextExtractor& other) const;

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & boost::serialization::base_object<FeatureContextExtractor>(*this);
    ar & classId;
    ar & hasher;
  }

  int classId;
  boost::shared_ptr<FeatureContextHasher> hasher;
};

} // namespace oxlm

BOOST_CLASS_EXPORT_KEY(oxlm::WordContextExtractor)
