#pragma once

#include <string>

#include <boost/shared_ptr.hpp>

// cdec headers
#include "ff.h"
#include "hg.h"
#include "sentence_metadata.h"

#include "lbl/cdec_conditional_state_converter.h"
#include "lbl/cdec_lbl_mapper.h"
#include "lbl/cdec_rule_converter.h"
#include "lbl/cdec_state_converter.h"
#include "lbl/lbl_features.h"
#include "lbl/model.h"
#include "lbl/query_cache.h"
#include "lbl/parallel_vocabulary.h"

using namespace std;

namespace oxlm {

struct AlignmentLinkInfo {
  int sourceIndex;
  int targetDistanceFromEdge;
};

struct AlignmentComparator {
  bool operator()(const AlignmentPoint& lhs, const AlignmentPoint& rhs) const {
    return lhs.t_ < rhs.t_ || (lhs.t_ == rhs.t_ && lhs.s_ < rhs.s_);
  }
};

class FF_SourceLBLLM : public FeatureFunction {
 public:
  FF_SourceLBLLM(
        const string& filename,
        const string& feature_name,
        bool normalized,
        bool persistent_cache);

  virtual void PrepareForInput(const SentenceMetadata& smeta);

  virtual ~FF_SourceLBLLM();

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

  const vector<WordID> GetSourceSentence(
      const SentenceMetadata& smeta) const;

  vector<int> findSourceSidePositions(
      const vector<int>& source,
      const vector<const void*>& prevStates,
      int spanStart,
      int spanEnd) const;

  int findAffiliation(const vector<int>& sourcePositions,
      const vector<int>& target,
      const vector<AlignmentPoint>& alignment,
      const vector<const void*>& prevStates,
      int targetWordIndex) const;

  AlignmentLinkInfo findLeftMostLink(
      const vector<int>& target,
      const vector<AlignmentPoint>& alignment,
      const vector<int>& affiliations,
      const vector<const void*>& prev_states) const;

  AlignmentLinkInfo findRightMostLink(
      const vector<int>& target,
      const vector<AlignmentPoint>& alignment,
      const vector<int>& affiliations,
      const vector<const void*>& prev_states) const;

  int countTerminalsCovered(
      int index,
      const vector<int>& target,
      const vector<const void*>& prev_states) const;

  int getTargetLength(
      const vector<int>& target, const vector<const void*>& prev_states) const;

  vector<int> convertAffiliations(
      const vector<int>& target,
      const vector<int>& affiliations,
      const vector<const void*>& prev_states) const;

  LBLFeatures scoreFullContexts(
      const vector<int>& symbols, const vector<int>& affiliations) const;

  Real getScore(int word, const vector<int>& context) const;

  LBLFeatures scoreContext(
      const vector<int>& symbols, int position, int affiliation) const;

  void prune(vector<int>& symbols, vector<int>& affiliations) const;

  void constructNextState(
      void* state, vector<int>& symbols, vector<int>& affiliations,
      int spanStart, int spanEnd, int targetLength,
      int leftMostLinkSource, int leftMostLinkDistance,
      int rightMostLinkSource, int rightMostLinkDistance ) const;

  LBLFeatures estimateScore(
      const vector<int>& symbols, const vector<int>& affiliations) const;

  int fid;
  int fidOOV;
  string filename;

  boost::shared_ptr<ParallelVocabulary> vocab;
  boost::shared_ptr<ModelData> config;
  SourceFactoredLM model;
  boost::shared_ptr<CdecLBLMapper> mapper;
  boost::shared_ptr<CdecRuleConverter> ruleConverter;
  boost::shared_ptr<CdecConditionalStateConverter> stateConverter;

  int kSTART;
  int kSTOP;
  int kUNKNOWN;
  int kSTAR;

  bool normalized;

  bool persistentCache;
  string cacheFile;
  mutable QueryCache cache;
  mutable int cacheHits, totalHits;

  vector<WordID> sourceSentence;
};

} // namespace oxlm
