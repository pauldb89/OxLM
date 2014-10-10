#include "lbl/cdec_ff_lbl.h"

#include <iostream>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>

using namespace std;
using namespace oxlm;

namespace oxlm {

template<class Model>
FF_LBLLM<Model>::FF_LBLLM(
    const string& filename,
    const string& feature_name,
    bool normalized,
    bool persistent_cache)
    : fid(FD::Convert(feature_name)),
      fidOOV(FD::Convert(feature_name + "_OOV")),
      filename(filename), normalized(normalized),
      persistentCache(persistent_cache), cacheHits(0), totalHits(0) {
  model.load(filename);

  config = model.getConfig();
  int context_width = config->ngram_order - 1;
  // For each state, we store at most context_width word ids to the left and
  // to the right and a kSTAR separator. The last bit represents the actual
  // size of the state.
  int max_state_size = (2 * context_width + 1) * sizeof(int) + 1;
  FeatureFunction::SetStateSize(max_state_size);

  vocab = model.getVocab();
  mapper = boost::make_shared<CdecLBLMapper>(vocab);
  stateConverter = boost::make_shared<CdecStateConverter>(max_state_size - 1);
  ruleConverter = boost::make_shared<CdecRuleConverter>(mapper, stateConverter);

  kSTART = vocab->convert("<s>");
  kSTOP = vocab->convert("</s>");
  kUNKNOWN = vocab->convert("<unk>");
  kSTAR = vocab->convert("<{STAR}>");
}

template<class Model>
void FF_LBLLM<Model>::savePersistentCache() {
  if (persistentCache && cacheFile.size()) {
    ofstream f(cacheFile);
    boost::archive::binary_oarchive oa(f);
    cerr << "Saving n-gram probability cache to " << cacheFile << endl;
    oa << cache;
    cerr << "Finished saving " << cache.size()
         << " n-gram probabilities..." << endl;
  }
}

template<class Model>
void FF_LBLLM<Model>::loadPersistentCache(int sentence_id) {
  if (persistentCache) {
    cacheFile = filename + "." + to_string(sentence_id) + ".cache.bin";
    if (boost::filesystem::exists(cacheFile)) {
      ifstream f(cacheFile);
      boost::archive::binary_iarchive ia(f);
      cerr << "Loading n-gram probability cache from " << cacheFile << endl;
      ia >> cache;
      cerr << "Finished loading " << cache.size()
           << " n-gram probabilities..." << endl;
    } else {
      cerr << "Cache file not found..." << endl;
    }
  }
}

template<class Model>
void FF_LBLLM<Model>::PrepareForInput(const SentenceMetadata& smeta) {
  model.clearCache();

  savePersistentCache();
  cache.clear();
  loadPersistentCache(smeta.GetSentenceId());
}

template<class Model>
void FF_LBLLM<Model>::TraversalFeaturesImpl(
    const SentenceMetadata& smeta, const HG::Edge& edge,
    const vector<const void*>& prev_states, SparseVector<double>* features,
    SparseVector<double>* estimated_features, void* next_state) const {
  vector<int> symbols = ruleConverter->convertTargetSide(
      edge.rule_->e(), prev_states);

  LBLFeatures exact_scores = scoreFullContexts(symbols);
  if (exact_scores.LMScore) {
    features->set_value(fid, exact_scores.LMScore);
  }
  if (exact_scores.OOVScore) {
    features->set_value(fidOOV, exact_scores.OOVScore);
  }

  constructNextState(symbols, next_state);
  symbols = stateConverter->getTerminals(next_state);

  LBLFeatures estimated_scores = estimateScore(symbols);
  if (estimated_scores.LMScore) {
    estimated_features->set_value(fid, estimated_scores.LMScore);
  }
  if (estimated_scores.OOVScore) {
    estimated_features->set_value(fidOOV, estimated_scores.OOVScore);
  }
}

template<class Model>
void FF_LBLLM<Model>::FinalTraversalFeatures(
    const void* prev_state, SparseVector<double>* features) const {
  vector<int> symbols = stateConverter->getTerminals(prev_state);
  symbols.insert(symbols.begin(), kSTART);
  symbols.push_back(kSTOP);

  LBLFeatures final_scores = estimateScore(symbols);
  if (final_scores.LMScore) {
    features->set_value(fid, final_scores.LMScore);
  }
  if (final_scores.OOVScore) {
    features->set_value(fidOOV, final_scores.OOVScore);
  }
}

template<class Model>
LBLFeatures FF_LBLLM<Model>::scoreFullContexts(
    const vector<int>& symbols) const {
  // Returns the sum of the scores of all the sequences of symbols other
  // than kSTAR that has length of at least ngram_order, score.
  LBLFeatures ret;
  int last_star = -1;
  int context_width = config->ngram_order - 1;
  for (size_t i = 0; i < symbols.size(); ++i) {
    if (symbols[i] == kSTAR) {
      last_star = i;
    } else if (i - last_star > context_width) {
      ret += scoreContext(symbols, i);
    }
  }

  return ret;
}

template<class Model>
Real FF_LBLLM<Model>::getScore(int word, const vector<int>& context) const {
  if (normalized) {
    return model.getLogProb(word, context);
  } else {
    return model.getUnnormalizedScore(word, context);
  }
}

template<class Model>
LBLFeatures FF_LBLLM<Model>::scoreContext(
    const vector<int>& symbols, int position) const {
  int word = symbols[position];
  int context_width = config->ngram_order - 1;

  // Push up to the last context_width words into the context vector.
  // Note that the most recent context word is first, so if we're
  // scoring the word "diplomatic" with a 4-gram context in the sentence
  // "Australia is one of the few countries with diplomatic relations..."
  // the context vector would be ["with", "countries", "few"].
  vector<int> context;
  for (int i = 1; i <= context_width && position - i >= 0; ++i) {
    assert(symbols[position - i] != kSTAR);
    context.push_back(symbols[position - i]);
  }

  // If we haven't filled the full context, then pad it.
  // If the context hits the <s>, then pad with more <s>'s.
  // Otherwise, if the context is short due to a kSTAR,
  // pad with UNKs.
  if (!context.empty() && context.back() == kSTART) {
    context.resize(context_width, kSTART);
  } else {
    context.resize(context_width, kUNKNOWN);
  }

  // Check the cache for this context.
  // If it's in there, use the saved values as score.
  // Otherwise, run the full model to get the score value.
  double score;
  if (persistentCache) {
    NGram query(word, context);
    ++totalHits;
    pair<double, bool> ret = cache.get(query);
    if (ret.second) {
      ++cacheHits;
      score = ret.first;
    } else {
      score = getScore(word, context);
      cache.put(query, score);
    }
  } else {
    score = getScore(word, context);
  }

  // Return the score, along with the OOV indicator feature value
  return LBLFeatures(score, word == kUNKNOWN);
}

template<class Model>
void FF_LBLLM<Model>::constructNextState(
    const vector<int>& symbols, void* state) const {
  int context_width = config->ngram_order - 1;

  vector<int> next_state;
  for (size_t i = 0; i < symbols.size() && i < context_width; ++i) {
    if (symbols[i] == kSTAR) {
      break;
    }
    next_state.push_back(symbols[i]);
  }

  if (next_state.size() < symbols.size()) {
    next_state.push_back(kSTAR);

    int last_star = -1;
    for (size_t i = 0; i < symbols.size(); ++i) {
      if (symbols[i] == kSTAR) {
        last_star = i;
      }
    }

    size_t i = max(last_star + 1, static_cast<int>(symbols.size() - context_width));
    while (i < symbols.size()) {
      next_state.push_back(symbols[i]);
      ++i;
    }
  }

  stateConverter->convert(state, next_state);
}

template<class Model>
LBLFeatures FF_LBLLM<Model>::estimateScore(const vector<int>& symbols) const {
  // Scores the symbols up to the first kSTAR, or up to the context_width,
  // whichever is first, padding the context with kSTART or kUNKNOWN as
  // needed. This offsets the fact that by scoreFullContexts() does not
  // score the first context_width words of a sentence.
  LBLFeatures ret = scoreFullContexts(symbols);

  int context_width = config->ngram_order - 1;
  for (size_t i = 0; i < symbols.size() && i < context_width; ++i) {
    if (symbols[i] == kSTAR) {
      break;
    }

    if (symbols[i] != kSTART) {
      ret += scoreContext(symbols, i);
    }
  }

  return ret;
}

template<class Model>
FF_LBLLM<Model>::~FF_LBLLM() {
  savePersistentCache();
  if (persistentCache) {
    cerr << "Cache hit ratio: " << 100.0 * cacheHits / totalHits
         << " %" << endl;
  }
}

template class FF_LBLLM<LM>;
template class FF_LBLLM<FactoredLM>;
template class FF_LBLLM<FactoredMaxentLM>;

} // namespace oxlm
