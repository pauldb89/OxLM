#pragma once

namespace oxlm {

/**
 * Wraps the feature values computed from the LBL language model.
 */
struct LBLFeatures {
  LBLFeatures();

  LBLFeatures(double lm_score, double oov_score);

  LBLFeatures& operator+=(const LBLFeatures& other);

  double LMScore;
  double OOVScore;
};

} // namespace oxlm
