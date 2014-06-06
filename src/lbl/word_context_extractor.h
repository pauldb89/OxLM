#pragma once

#include <boost/serialization/shared_ptr.hpp>

#include "lbl/archive_export.h"
#include "lbl/feature_context_extractor.h"
#include "lbl/feature_context_mapper.h"

namespace oxlm {

class WordContextExtractor : public FeatureContextExtractor {
 public:
  WordContextExtractor();

  WordContextExtractor(
      int class_id, const boost::shared_ptr<FeatureContextMapper>& mapper);

  virtual vector<int> getFeatureContextIds(const vector<int>& context) const;

  int getFeatureContextId(const FeatureContext& feature_context) const;

  bool operator==(const WordContextExtractor& other) const;

  virtual ~WordContextExtractor();

 private:
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & boost::serialization::base_object<FeatureContextExtractor>(*this);
    ar & classId;
    ar & mapper;
  }

  int classId;
  boost::shared_ptr<FeatureContextMapper> mapper;
};

} // namespace oxlm

BOOST_CLASS_EXPORT_KEY(oxlm::WordContextExtractor)
