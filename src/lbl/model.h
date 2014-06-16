#pragma once

#include <boost/shared_ptr.hpp>

#include "corpus/corpus.h"
#include "lbl/config.h"
#include "lbl/utils.h"

namespace oxlm {

template<class GlobalWeights, class MinibatchWeights, class Metadata>
class Model {
 public:
  Model(ModelData& config);

  void learn();

  void update(
      const boost::shared_ptr<MinibatchWeights>& global_gradient,
      const boost::shared_ptr<GlobalWeights>& adagrad);

  Real regularize(Real minibatch_factor);

  void evaluate(
      const boost::shared_ptr<Corpus>& corpus, const Time& iteration_start,
      int minibatch_counter, Real& objective, Real& best_perplexity) const;

  void save() const;

 private:
  ModelData config;
  Dict dict;
  boost::shared_ptr<Metadata> metadata;
  boost::shared_ptr<GlobalWeights> weights;
};

} // namespace oxlm
