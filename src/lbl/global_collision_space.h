#pragma once

#include <boost/serialization/array.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/split_member.hpp>

#include "lbl/utils.h"

namespace oxlm {

class GlobalCollisionSpace {
 public:
  GlobalCollisionSpace();

  GlobalCollisionSpace(const GlobalCollisionSpace& other);

  GlobalCollisionSpace(int hash_space_size);

  GlobalCollisionSpace& operator=(const GlobalCollisionSpace& other);

  bool operator==(const GlobalCollisionSpace& other) const;

  virtual ~GlobalCollisionSpace();

 private:
  void deepCopy(const GlobalCollisionSpace& other);

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
