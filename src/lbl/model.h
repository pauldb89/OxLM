#pragma once

#include "lbl/global_weights.h"
#include "lbl/global_factored_weights.h"
#include "lbl/global_factored_maxent_weights.h"
#include "lbl/minibatch_weights.h"
#include "lbl/minibatch_factored_weights.h"
#include "lbl/minibatch_factored_maxent_weights.h"
#include "lbl/metadata.h"
#include "lbl/factored_metadata.h"
#include "lbl/factored_maxent_metadata.h"

namespace oxlm {

template<class GlobalWeights, class MinibatchWeights, class Metadata>
class Model {
 public:
  void learn();

  void computeGradient() const;

  void regularize();

  void evaluate();

 private:
  GlobalWeights weights;
  Metadata metadata;
};

class NLM : public Model<GlobalWeights, MinibatchWeights, Metadata> {};
class FactoredNLM : public Model<GlobalFactoredWeights, MinibatchFactoredWeights, FactoredMetadata> {};
class FactoredMaxentNLM : public Model<GlobalFactoredMaxentWeights, MinibatchFactoredMaxentWeights, FactoredMaxentMetadata> {};

} // namespace oxlm
