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

  void computeGradient(
      const boost::shared_ptr<Corpus>& corpus,
      const vector<int>& indices,
      boost::shared_ptr<MinibatchWeights>& gradient,
      Real& objective) const;

  void regularize();

  void evaluate();

 private:
  ModelData config;
  Dict dict;
  boost::shared_ptr<Metadata> metadata;
  boost::shared_ptr<GlobalWeights> weights;
};

} // namespace oxlm
