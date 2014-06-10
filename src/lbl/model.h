#pragma once

#include <boost/shared_ptr.hpp>

#include "corpus/corpus.h"
#include "lbl/config.h"
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
  Model(const ModelData& config);

  void learn();

  void computeGradient() const;

  void regularize();

  void evaluate();

 private:
  ModelData config;
  Dict dict;
  boost::shared_ptr<Metadata> metadata;
  boost::shared_ptr<GlobalWeights> weights;
};

} // namespace oxlm
