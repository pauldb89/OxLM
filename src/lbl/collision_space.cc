#include "lbl/collision_space.h"

namespace oxlm {

CollisionSpace::CollisionSpace() {}

CollisionSpace::CollisionSpace(const CollisionSpace& other) {
  deepCopy(other);
}

CollisionSpace::CollisionSpace(int hash_space_size)
    : hashSpaceSize(hash_space_size) {
  featureWeights = new Real[hashSpaceSize];
  VectorRealMap featureWeightsMap(featureWeights, hashSpaceSize);
  featureWeightsMap = VectorReal::Zero(hashSpaceSize);
}

CollisionSpace& CollisionSpace::operator=(const CollisionSpace& other) {
  deepCopy(other);
  return *this;
}

bool CollisionSpace::operator==(const CollisionSpace& other) const {
  if (hashSpaceSize != other.hashSpaceSize) {
    return true;
  }
}

void CollisionSpace::deepCopy(const CollisionSpace& other) {
  hashSpaceSize = other.hashSpaceSize;

  featureWeights = new Real[hashSpaceSize];
  memcpy(featureWeights, other.featureWeights, hashSpaceSize * sizeof(Real));
}

CollisionSpace::~CollisionSpace() {
  delete featureWeights;
}

} // namespace oxlm
