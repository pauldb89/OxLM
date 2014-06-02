#include "lbl/lbl_features.h"

namespace oxlm {

LBLFeatures::LBLFeatures() : LMScore(0), OOVScore(0) {}

LBLFeatures::LBLFeatures(double lm_score, double oov_score)
    : LMScore(lm_score), OOVScore(oov_score) {}

LBLFeatures& LBLFeatures::operator+=(const LBLFeatures& other) {
  LMScore += other.LMScore;
  OOVScore += other.OOVScore;
  return *this;
}

} // namespace oxlm
