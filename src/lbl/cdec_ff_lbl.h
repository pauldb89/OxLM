#pragma once

#include <string>

#include <boost/shared_ptr.hpp>

// cdec headers
#include "ff.h"
#include "hg.h"
#include "sentence_metadata.h"

#include "lbl/cdec_lbl_mapper.h"
#include "lbl/cdec_rule_converter.h"
#include "lbl/cdec_state_converter.h"
#include "lbl/lbl_features.h"
#include "lbl/model.h"
#include "lbl/query_cache.h"

namespace oxlm {

template<class Model>
class FF_LBLLM : public FeatureFunction {
 public:
  FF_LBLLM(
      const string& filename,
      const string& feature_name,
      bool normalized,
      bool persistent_cache);

  virtual void PrepareForInput(const SentenceMetadata& smeta);

  ~FF_LBLLM();

 protected:
  virtual void TraversalFeaturesImpl(
      const SentenceMetadata& smeta, const HG::Edge& edge,
      const vector<const void*>& prev_states, SparseVector<double>* features,
      SparseVector<double>* estimated_features, void* next_state) const;

  virtual void FinalTraversalFeatures(
      const void* prev_state, SparseVector<double>* features) const;

 private:
  void savePersistentCache();

  void loadPersistentCache(int sentence_id);

  LBLFeatures scoreFullContexts(const vector<int>& symbols) const;

  Real getScore(int word, const vector<int>& context) const;

  LBLFeatures scoreContext(const vector<int>& symbols, int position) const;

  void constructNextState(const vector<int>& symbols, void* state) const;

  LBLFeatures estimateScore(const vector<int>& symbols) const;

  int fid;
  int fidOOV;
  string filename;

  boost::shared_ptr<Vocabulary> vocab;
  boost::shared_ptr<ModelData> config;
  Model model;
  boost::shared_ptr<CdecLBLMapper> mapper;
  boost::shared_ptr<CdecRuleConverter> ruleConverter;
  boost::shared_ptr<CdecStateConverter> stateConverter;

  int kSTART;
  int kSTOP;
  int kUNKNOWN;
  int kSTAR;

  bool normalized;

  bool persistentCache;
  string cacheFile;
  mutable QueryCache cache;
  mutable int cacheHits, totalHits;
};

} // namespace oxlm
