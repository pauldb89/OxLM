#include "lbl/global_collision_space.h"

#include "utils/constants.h"

namespace oxlm {

GlobalCollisionSpace::GlobalCollisionSpace() {}

GlobalCollisionSpace::GlobalCollisionSpace(const GlobalCollisionSpace& other) {
  deepCopy(other);
}

GlobalCollisionSpace::GlobalCollisionSpace(int hash_space_size)
    : hashSpaceSize(hash_space_size) {
  featureWeights = new Real[hashSpaceSize];
  VectorRealMap featureWeightsMap(featureWeights, hashSpaceSize);
  featureWeightsMap = VectorReal::Zero(hashSpaceSize);
}

GlobalCollisionSpace& GlobalCollisionSpace::operator=(const GlobalCollisionSpace& other) {
  deepCopy(other);
  return *this;
}

bool GlobalCollisionSpace::operator==(const GlobalCollisionSpace& other) const {
  if (hashSpaceSize != other.hashSpaceSize) {
    return true;
  }
}

void GlobalCollisionSpace::deepCopy(const GlobalCollisionSpace& other) {
  hashSpaceSize = other.hashSpaceSize;

  featureWeights = new Real[hashSpaceSize];
  memcpy(featureWeights, other.featureWeights, hashSpaceSize * sizeof(Real));
}

GlobalCollisionSpace::~GlobalCollisionSpace() {
  delete featureWeights;
}

} // namespace oxlm
