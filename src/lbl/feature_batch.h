#pragma once

#include "lbl/utils.h"

namespace oxlm {

class FeatureBatch {
 public:
  virtual VectorReal values() const = 0;

  virtual void add(const VectorReal& values) = 0;

  virtual void setZero() = 0;
};

} // namespace oxlm
