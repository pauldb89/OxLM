#pragma once

#include <boost/shared_ptr.hpp>

#include "corpus/corpus.h"
#include "lbl/config.h"
#include "lbl/factored_metadata.h"
#include "lbl/factored_maxent_metadata.h"
#include "lbl/factored_weights.h"
#include "lbl/global_factored_maxent_weights.h"
#include "lbl/metadata.h"
#include "lbl/minibatch_factored_maxent_weights.h"
#include "lbl/model_utils.h"
#include "lbl/utils.h"
#include "lbl/weights.h"

namespace oxlm {

enum ModelType {
  NLM = 1,
  FACTORED_NLM = 2,
  FACTORED_MAXENT_NLM = 3,
};

template<class GlobalWeights, class MinibatchWeights, class Metadata>
class Model {
 public:
  Model();

  Model(const boost::shared_ptr<ModelData>& config);

  Dict getDict() const;

  boost::shared_ptr<ModelData> getConfig() const;

  void learn();

  void update(
      const boost::shared_ptr<MinibatchWeights>& global_gradient,
      const boost::shared_ptr<GlobalWeights>& adagrad);

  Real regularize(
      const boost::shared_ptr<MinibatchWeights>& global_gradient,
      Real minibatch_factor);

  void evaluate(
      const boost::shared_ptr<Corpus>& corpus, Real& accumulator) const;

  Real predict(int word_id, const vector<int>& context) const;

  MatrixReal getWordVectors() const;

  void save() const;

  void load(const string& filename);

  void clearCache();

  bool operator==(
      const Model<GlobalWeights, MinibatchWeights, Metadata>& other) const;

 private:
  void evaluate(
      const boost::shared_ptr<Corpus>& corpus, const Time& iteration_start,
      int minibatch_counter, Real& objective, Real& best_perplexity) const;

  boost::shared_ptr<ModelData> config;
  Dict dict;
  boost::shared_ptr<Metadata> metadata;
  boost::shared_ptr<GlobalWeights> weights;
};

class LM : public Model<Weights, Weights, Metadata> {
 public:
  LM() : Model<Weights, Weights, Metadata>() {}

  LM(const boost::shared_ptr<ModelData>& config)
      : Model<Weights, Weights, Metadata>(config) {}
};

class FactoredLM: public Model<FactoredWeights, FactoredWeights, FactoredMetadata> {
 public:
  FactoredLM() : Model<FactoredWeights, FactoredWeights, FactoredMetadata>() {}

  FactoredLM(const boost::shared_ptr<ModelData>& config)
      : Model<FactoredWeights, FactoredWeights, FactoredMetadata>(config) {}
};

class FactoredMaxentLM : public Model<GlobalFactoredMaxentWeights, MinibatchFactoredMaxentWeights, FactoredMaxentMetadata> {
 public:
  FactoredMaxentLM() : Model<GlobalFactoredMaxentWeights, MinibatchFactoredMaxentWeights, FactoredMaxentMetadata>() {}

  FactoredMaxentLM(const boost::shared_ptr<ModelData>& config)
      : Model<GlobalFactoredMaxentWeights, MinibatchFactoredMaxentWeights, FactoredMaxentMetadata>(config) {}
};

} // namespace oxlm
