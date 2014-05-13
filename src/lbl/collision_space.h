#pragma once

#include <boost/serialization/array.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_member.hpp>

#include "lbl/utils.h"

namespace oxlm {

class CollisionSpace {
 public:
  CollisionSpace();

  CollisionSpace(const CollisionSpace& other);

  CollisionSpace(int hash_space_size);

  CollisionSpace& operator=(const CollisionSpace& other);

  bool operator==(const CollisionSpace& other) const;

  virtual ~CollisionSpace();

 private:
  void deepCopy(const CollisionSpace& other);

  friend class boost::serialization::access;

  template<class Archive>
  void save(Archive& ar, const unsigned int version) const {
    ar << hashSpaceSize;
    ar << boost::serialization::make_array(featureWeights, hashSpaceSize);
  }

  template<class Archive>
  void load(Archive& ar, const unsigned int version) {
    ar >> hashSpaceSize;
    featureWeights = new Real[hashSpaceSize];
    ar >> boost::serialization::make_array(featureWeights, hashSpaceSize);
  }

  BOOST_SERIALIZATION_SPLIT_MEMBER();

  friend class CollisionGlobalFeatureStore;

  int hashSpaceSize;
  Real* featureWeights;
};

} // namespace oxlm
