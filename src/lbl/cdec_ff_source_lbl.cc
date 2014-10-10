#include "lbl/cdec_ff_source_lbl.h"

#include <iostream>
#include <string>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/filesystem.hpp>
#include <boost/make_shared.hpp>

#include "lbl/cdec_conditional_state_converter.h"
#include "lbl/cdec_lbl_mapper.h"
#include "lbl/cdec_rule_converter.h"
#include "lbl/cdec_state_converter.h"
#include "lbl/lbl_features.h"
#include "lbl/model.h"
#include "lbl/query_cache.h"
#include "lbl/parallel_vocabulary.h"

using namespace oxlm;

namespace oxlm {

FF_SourceLBLLM::FF_SourceLBLLM(
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
  // For each NT, we store the following state information:
  // - Up to context_width target word ids to the left and
  //   to the right and a kSTAR separator. (2 * n + 1)
  // - The affiliation of each target word. (2 * n + 1)
  // - The absolute index of the source side of the left and rightmost
  //   alignment links covered by the NT (2)
  // - The distance from the left/right edge to the target terminal of
  //   the left/rightmost alignment link covered by the NT, measured in
  //   target terminals (2)
  // - Two integers representing the source sentence span covered (2)
  // - One integer representing the size of the target side of the NT (1)
  // - The last byte represents the actual size of the state.
  // Total: (4 * context_width + 9) int's + 1 char
  int max_state_size = (4 * context_width + 9) * sizeof(int) + 1;
  FeatureFunction::SetStateSize(max_state_size);

  vocab = dynamic_pointer_cast<ParallelVocabulary>(model.getVocab());
  mapper = boost::make_shared<CdecLBLMapper>(vocab);
  stateConverter = boost::make_shared<CdecConditionalStateConverter>(max_state_size - 1);
  ruleConverter = boost::make_shared<CdecRuleConverter>(mapper, stateConverter);

  kSTART = vocab->convert("<s>");
  kSTOP = vocab->convert("</s>");
  kUNKNOWN = vocab->convert("<unk>");
  kSTAR = vocab->convert("<{STAR}>");
}

void FF_SourceLBLLM::savePersistentCache() {
  if (persistentCache && cacheFile.size()) {
    ofstream f(cacheFile);
    boost::archive::binary_oarchive oa(f);
    cerr << "Saving n-gram probability cache to " << cacheFile << endl;
    oa << cache;
    cerr << "Finished saving " << cache.size()
         << " n-gram probabilities..." << endl;
  }
}

void FF_SourceLBLLM::loadPersistentCache(int sentence_id) {
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

// The function extracts the source sentence from the SentenceMetadata,
// and converts it from cdec word id space to OxLM word id space.
const vector<WordID> FF_SourceLBLLM::GetSourceSentence(
    const SentenceMetadata& smeta) const {
  vector<WordID> ret;
  const Lattice& lattice = smeta.GetSourceLattice();
  for (int i = 0; i < lattice.size(); ++i) {
    if (lattice[i].size() > 1) {
      cerr << "OxLM with source side conditioning "
           << "does not support lattice input.\n";
      exit(1);
    }

    int cdec_label = lattice[i][0].label;
    string word = TD::Convert(cdec_label);
    int oxlm_label = vocab->convertSource(word, true);
    if (oxlm_label == -1) {
      oxlm_label = vocab->convertSource("<unk>");
    }
    ret.push_back(oxlm_label);
  }

  return ret;
}

void FF_SourceLBLLM::PrepareForInput(const SentenceMetadata& smeta) {
  model.clearCache();

  savePersistentCache();
  cache.clear();
  loadPersistentCache(smeta.GetSentenceId());

  this->sourceSentence = GetSourceSentence(smeta);
}

// Returns a vector of ints the same size as source that represents
// the (start) position of each symbol in the full source sentence.
// Note that for terminals, their position is equal to their start position.
vector<int> FF_SourceLBLLM::findSourceSidePositions(
    const vector<int>& source,
    const vector<const void*>& prevStates,
    int spanStart,
    int spanEnd) const {
  vector<int> positions;
  vector<pair<int, int>> nt_spans;

  for (int i = 0; i < prevStates.size(); ++i) {
    int start = stateConverter->getSourceSpanStart(prevStates[i]);
    int end = stateConverter->getSourceSpanEnd(prevStates[i]);
    nt_spans.push_back(make_pair(start, end));
  }
  sort(nt_spans.begin(), nt_spans.end());

  int i = spanStart; // Index into the source sentence
  int j = 0; // Index into the source vector (of a particular hyperedge)
  int ntcount = 0; // The number of NTs we've seen so far in this hyperedge
  while (i < spanEnd) {
    assert (j < source.size());
    positions.push_back(i);

    // If this symbol is a non-terminal
    if (source[j] <= 0) {
      assert (ntcount < prevStates.size());
      assert (get<0>(nt_spans[ntcount]) == i);
      const pair<int, int>& span = nt_spans[ntcount];
      int ntLength = get<1>(span) - get<0>(span);
      // Jump past the span covered by the NT
      i += ntLength;
      ++ntcount;
    } else {
      // All terminals have length 1.
      i += 1;
    }
    ++j;
  }
  assert (positions.size() == source.size());
  return positions;
}

// Finds the "affiliation" of a target word.
// See "Fast and Robust Neural Network Joint Models for Statistical
// Machine Translation" (Devlin et al. 2014) for more information.
// sourcePositions is the output of findSourcePositions()
// target is rule->e()
// alignment is rule->als().
// prev_states is the same as what's passed in to the traverse function.
// targetWordIndex is an index into rule->e(), NOT the target sentence.
// Return value is an index into the whole source sentence.
int FF_SourceLBLLM::findAffiliation(
    const vector<int>& sourcePositions,
    const vector<int>& target,
    const vector<AlignmentPoint>& alignment,
    const vector<const void*>& prevStates,
    int targetWordIndex) const {
  assert (targetWordIndex < target.size());
  assert (target[targetWordIndex] > 0);
  vector<vector<int>> alignment_by_index(target.size());
  for (AlignmentPoint ap : alignment) {
    alignment_by_index[ap.t_].push_back(ap.s_);
  }

  vector<const void*> state_by_index(target.size(), NULL);
  for (int i = 0; i < target.size(); ++i) {
    if (target[i] <= 0) {
      state_by_index[i] = prevStates[-target[i]];
    }
  }

  // If the target word at the desired index is aligned
  // we can short circuit and just return the word it's aligned to.
  // If there's more than one, we return the middle one.
  if (alignment_by_index[targetWordIndex].size() != 0) {
    int middle = (alignment_by_index[targetWordIndex].size() - 1) / 2;
    return sourcePositions[alignment_by_index[targetWordIndex][middle]];
  }

  int left_index = targetWordIndex - 1;
  int left_offset = 1;
  int right_index = targetWordIndex + 1;
  int right_offset = 1;

  // We iteratively look at a word either to the left or right of the target word,
  // always looking at the terminal closest to the target terminal. In the case
  // that we haven't looked at a word to the right and a word to the left that are
  // both equidistant from the target, we arbitrarily look to the right.
  while (left_index >= 0 || right_index < target.size()) {
    if (left_index >= 0 && (left_offset < right_offset || right_index == target.size())) {
      // If we want to look at the word at target[targetWordIndex - left_offset]
      if (alignment_by_index[left_index].size() != 0) {
        // Aligned terminal
        assert (target[left_index] > 0);
        // If this word has multiple alignments, we want the middle one.
        vector<int>& alignments = alignment_by_index[left_index];
        int a = alignments[(alignments.size() - 1) / 2];
        return sourcePositions[a];
      } else if (target[left_index] <= 0) {
        const void* state = state_by_index[left_index];
        assert (state != NULL);
        int rightmost_link_source = stateConverter->getRightmostLinkSource(state);
        if (rightmost_link_source != -1) {
          // Aligned non-terminal
          return rightmost_link_source;
        } else {
          // Unaligned non-terminal
          left_offset += stateConverter->getTargetSpanSize(state);
          left_index--;
        }
      } else {
        // Unaligned terminal
        left_offset++;
        left_index--;
      }
    } else {
      // Otherwise we want to look at the word at target[targetWordIndex + right_offset + 1]
      if (alignment_by_index[right_index].size() != 0) {
        // Aligned terminal
        assert (target[right_index] > 0);
        // If this word has multiple alignments, we want the middle one.
        vector<int>& alignments = alignment_by_index[right_index];
        int a = alignments[(alignments.size() - 1) / 2];
        return sourcePositions[a];
      } else if (target[right_index] <= 0) {
        const void* state = state_by_index[right_index];
        assert (state != NULL);
        int leftmost_link_source = stateConverter->getLeftmostLinkSource(state);
        if (leftmost_link_source != -1) {
          // Aligned non-terminal
          return leftmost_link_source;
        } else {
          // Unaligned non-terminal
          right_offset += stateConverter->getTargetSpanSize(state);
          right_index++;
        }
      } else {
        // Unaligned terminal
        right_offset++;
        right_index++;
      }
    }
  }

  // No alignment links were found anywhere among the target words covered
  // by this rule.
  return -1;
}

// Finds the left-most alignment link covered by a rule.
// The return value is a little struct that contains an index into the
// whole source sentence, along with a "distance" from the left side of
// this rule. If the first terminal covered by this rule is aligned
// the distance will be 0. Otherwise it will be the number of target
// terminals we must skip to find the first aligned target terminal.
AlignmentLinkInfo FF_SourceLBLLM::findLeftMostLink(
    const vector<int>& target,
    const vector<AlignmentPoint>& alignment,
    const vector<int>& affiliations,
    const vector<const void*>& prev_states) const {
  vector<AlignmentPoint> sorted_alignment(alignment);
  sort(sorted_alignment.begin(), sorted_alignment.end(), AlignmentComparator());

  AlignmentLinkInfo ret;
  int leftMostAlignedTerminal = (sorted_alignment.size() > 0) ? sorted_alignment[0].t_ : target.size();

  // There may be an NT covering an alignment link before the first
  // aligned terminal of this rule. Loop over them and check.
  // If we find one, we can return it right away.
  for (int i = 0; i < leftMostAlignedTerminal; ++i) {
    if (target[i] <= 0) {
      const void* state = prev_states[-target[i]];
      int nt_leftmost_link_source = stateConverter->getLeftmostLinkSource(state);
      if (nt_leftmost_link_source != -1) {
        int offset = stateConverter->getLeftmostLinkDistance(state);
        ret.sourceIndex = nt_leftmost_link_source;
        ret.targetDistanceFromEdge = countTerminalsCovered(i, target, prev_states) + offset;
        return ret;
      }
    }
  }

  // If we get to this point, then the leftmost terminal really is the leftmost alignment link.
  if (leftMostAlignedTerminal == target.size()) {
    // In this case there was no alignment links at all.
    ret.sourceIndex = -1;
    ret.targetDistanceFromEdge = -1;
    return ret;
  } else {
    ret.sourceIndex = affiliations[leftMostAlignedTerminal];
    ret.targetDistanceFromEdge = countTerminalsCovered(leftMostAlignedTerminal, target, prev_states);
    return ret;
  }
}

// Finds the right-most alignment link covered by a rule.
// The return value is a little struct that contains an index into the
// whole source sentence, along with a "distance" from the right side of
// this rule. If the last terminal covered by this rule is aligned
// the distance will be 0. Otherwise it will be the number of target
// terminals we must skip to find the lastaligned target terminal.
AlignmentLinkInfo FF_SourceLBLLM::findRightMostLink(
    const vector<int>& target,
    const vector<AlignmentPoint>& alignment,
    const vector<int>& affiliations,
    const vector<const void*>& prev_states) const {
  vector<AlignmentPoint> sorted_alignment(alignment);
  // Sort the alignment links in reverse this time.
  sort(sorted_alignment.rbegin(), sorted_alignment.rend(), AlignmentComparator());

  AlignmentLinkInfo ret;
  int rightMostAlignedTerminal = (sorted_alignment.size() > 0) ? sorted_alignment[0].t_ : -1;

  // There may be an NT covering an alignment link after the last
  // aligned terminal of this rule. Loop over them and check.
  // If we find one, we can return it right away.
  for (int i = target.size() - 1; i > rightMostAlignedTerminal; --i) {
    if (target[i] <= 0) {
      const void* state = prev_states[-target[i]];
      int nt_rightmost_link_source = stateConverter->getRightmostLinkSource(state);
      if (nt_rightmost_link_source != -1) {
        int offset = stateConverter->getRightmostLinkDistance(state);
        ret.sourceIndex = nt_rightmost_link_source;
        // length of span - (amount of stuff that came before us + our length) = amount of stuff after us
        ret.targetDistanceFromEdge = getTargetLength(target, prev_states)
          - countTerminalsCovered(i + 1, target, prev_states) + offset;
        return ret;
      }
    }
  }

  // If we get to this point, then the rightmost terminal really is the rightmost alignment link.
  if (rightMostAlignedTerminal == -1) {
    // In this case there was no alignment links at all.
    ret.sourceIndex = -1;
    ret.targetDistanceFromEdge = -1;
    return ret;
  }
  else {
    ret.sourceIndex = affiliations[sorted_alignment[0].t_];
    ret.targetDistanceFromEdge = getTargetLength(target, prev_states)
      - countTerminalsCovered(rightMostAlignedTerminal + 1, target, prev_states);
    return ret;
  }
}

// Counts the number of target terminals covered by a rule, up to (but not
// including) a given index into the target vector. If you want the number
//  of target terminals covered by the whole rule, set index=target.size().
int FF_SourceLBLLM::countTerminalsCovered(
    int index,
    const vector<int>& target,
    const vector<const void*>& prev_states) const {
  int targetLength = 0;

  for (int i = 0; i < index; ++i) {
    // If the current item is a terminal, it counts as one word
    if (target[i] > 0) {
      targetLength += 1;
    }
    // Otherwise it's an NT, and its length is stored in its corresponding
    // entry in prev_states.
    else {
      const void* state = prev_states[-target[i]];
      int nt_length = stateConverter->getTargetSpanSize(state);
      targetLength += nt_length;
    }
  }

  return targetLength;
}

// Finds the total number of target words covered by a rule.
// Note that this != target.size(), since some of the elements in
// target could be non-terminals that cover multiple words.
int FF_SourceLBLLM::getTargetLength(
    const vector<int>& target, const vector<const void*>& prev_states) const {
  return countTerminalsCovered(target.size(), target, prev_states);
}

// Converts the target side of a rule to a list of affiliations, one per
// covered target terminal. This unrolls NTs into the covered terminals
// before looking up affiliations.
vector<int> FF_SourceLBLLM::convertAffiliations(
    const vector<int>& target,
    const vector<int>& affiliations,
    const vector<const void*>& prev_states) const {
  assert (target.size() == affiliations.size());
  vector<int> ret;
  for (int i = 0; i < target.size(); ++i) {
    int symbol = target[i];
    if (symbol <= 0) {
      const void* state = prev_states[-symbol];
      const vector<int>& nt_affiliations = stateConverter->getAffiliations(state);
      ret.insert(ret.end(), nt_affiliations.begin(), nt_affiliations.end());
    }
    else {
      ret.push_back(affiliations[i]);
    }
  }
  return ret;
}

// This is the main function for the feature. It's called once for every hyperedge
// crossed during search.
void FF_SourceLBLLM::TraversalFeaturesImpl(
    const SentenceMetadata& smeta, const HG::Edge& edge,
    const vector<const void*>& prev_states, SparseVector<double>* features,
    SparseVector<double>* estimated_features, void* next_state) const {

  const vector<int>& source = edge.rule_->f();
  const vector<int>& target = edge.rule_->e();
  const vector<AlignmentPoint>& alignment = edge.rule_->als();

  const vector<int>& sourcePositions = findSourceSidePositions(
    source, prev_states, edge.i_, edge.j_);

  // Local affiliations: this doesn't account for terminals covered by
  // non-terminals in the local edge.
  vector<int> affiliations(target.size(), -1);
  for (int i = 0; i < target.size(); ++i) {
    if (target[i] > 0) {
     affiliations[i] = findAffiliation(sourcePositions, target, alignment, prev_states, i);
    }
  }

  // This unrolls non-terminals, giving us a list of all the terminals covered
  // by this rule, including those under NTs.
  vector<int> symbols = ruleConverter->convertTargetSide(target, prev_states);

  // Work out the affiliations for all the covered terminals, including
  // terminals covered by child NTs.
  vector<int> symbol_affiliations =
      convertAffiliations(target, affiliations, prev_states);
  assert (symbols.size() == symbol_affiliations.size());

  // Here's where we actually start making calls out to the neural net
  LBLFeatures exact_scores = scoreFullContexts(symbols, symbol_affiliations);
  if (exact_scores.LMScore) {
    features->set_value(fid, exact_scores.LMScore);
  }
  if (exact_scores.OOVScore) {
    features->set_value(fidOOV, exact_scores.OOVScore);
  }

  // Gather up a bunch of info needed to create the state that will be stored
  // for this NT.
  int targetLength = getTargetLength(target, prev_states);
  AlignmentLinkInfo leftmost =
      findLeftMostLink(target, alignment, affiliations, prev_states);
  AlignmentLinkInfo rightmost =
      findRightMostLink(target, alignment, affiliations, prev_states);
  assert(0 <= leftmost.sourceIndex && leftmost.sourceIndex < sourceSentence.size());
  assert(0 <= rightmost.sourceIndex && rightmost.sourceIndex < sourceSentence.size());

  constructNextState(
      next_state, symbols, symbol_affiliations,
      edge.i_, edge.j_, targetLength,
      leftmost.sourceIndex, leftmost.targetDistanceFromEdge,
      rightmost.sourceIndex, rightmost.targetDistanceFromEdge);

  // Convert the next state into a list of terminals, and use it to do
  // future cost estimation.
  LBLFeatures estimated_scores = estimateScore(symbols, symbol_affiliations);
  if (estimated_scores.LMScore) {
    estimated_features->set_value(fid, estimated_scores.LMScore);
  }
  if (estimated_scores.OOVScore) {
    estimated_features->set_value(fidOOV, estimated_scores.OOVScore);
  }
}

void FF_SourceLBLLM::FinalTraversalFeatures(
    const void* prev_state, SparseVector<double>* features) const {
  // When we have a full sentence we evaluate one more time, this time
  // surrounding the sentence with <s> and </s>.
  // Note: <s> and </s> are always affiliated with the first and last
  // words of the source sentence, respectively.
  vector<int> symbols = stateConverter->getTerminals(prev_state);
  symbols.insert(symbols.begin(), kSTART);
  symbols.push_back(kSTOP);

  vector<int> symbol_affiliations = stateConverter->getAffiliations(prev_state);
  symbol_affiliations.insert(symbol_affiliations.begin(), 0);
  symbol_affiliations.push_back(sourceSentence.size() - 1);

  LBLFeatures final_scores = estimateScore(symbols, symbol_affiliations);
  if (final_scores.LMScore) {
    features->set_value(fid, final_scores.LMScore);
  }
  if (final_scores.OOVScore) {
    features->set_value(fidOOV, final_scores.OOVScore);
  }
}

// Returns the sum of the scores of all the sequences of symbols other
// than kSTAR that has length of at least ngram_order.
LBLFeatures FF_SourceLBLLM::scoreFullContexts(
    const vector<int>& symbols, const vector<int>& affiliations) const {
  assert (symbols.size() == affiliations.size());
  LBLFeatures ret;
  int last_star = -1;
  int context_width = config->ngram_order - 1;
  for (size_t i = 0; i < symbols.size(); ++i) {
    if (symbols[i] == kSTAR) {
      last_star = i;
    } else if (i - last_star > context_width) {
      ret += scoreContext(symbols, i, affiliations[i]);
    }
  }

  return ret;
}

Real FF_SourceLBLLM::getScore(int word, const vector<int>& context) const {
  if (normalized) {
    return model.getLogProb(word, context);
  } else {
    return model.getUnnormalizedScore(word, context);
  }
}

// Scores an individual source word, given its context and affiliation
LBLFeatures FF_SourceLBLLM::scoreContext(
    const vector<int>& symbols, int position, int affiliation) const {
  int word = symbols[position];
  int context_width = config->ngram_order - 1;
  int source_context_width = 2 * config->source_order - 1;

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

  assert (context.size() == context_width);
  if (affiliation < 0 || affiliation >= sourceSentence.size()){
    cerr << "ERROR: Target word \"" << vocab->convert(symbols[position])
         << "\" has an affiliation of " << affiliation << ".";
    cerr << " Source sentence has length " << sourceSentence.size() << endl;
    assert (affiliation >= 0 && affiliation < sourceSentence.size());
  }

  // Extract the words in the source window, padding with <s> and </s>
  // as necessary.
  for (int i = affiliation - config->source_order + 1; i < affiliation + config->source_order; ++i) {
    if (i < 0) {
      context.push_back(vocab->convertSource("<s>"));
    }
    else if (i >= sourceSentence.size()) {
      context.push_back(vocab->convertSource("</s>"));
    }
    else {
      assert (sourceSentence[i] >= 0);
      context.push_back(sourceSentence[i]);
    }
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

// Removes extaneous symbols and affiliations from the given lists,
// leaving only up to context_width symbols and affiliations remaining on
// each side of the kSTAR symbol.
void FF_SourceLBLLM::prune(
    vector<int>& symbols, vector<int>& affiliations) const {
  assert (symbols.size() == affiliations.size());
  int context_width = config->ngram_order - 1;

  vector<int> pruned_symbols, pruned_affiliations;
  for (size_t i = 0; i < symbols.size() && i < context_width; ++i) {
    if (symbols[i] == kSTAR) {
      break;
    }
    pruned_symbols.push_back(symbols[i]);
    pruned_affiliations.push_back(affiliations[i]);
  }

  if (pruned_symbols.size() < symbols.size()) {
    pruned_symbols.push_back(kSTAR);
    pruned_affiliations.push_back(-1);

    int last_star = -1;
    for (size_t i = 0; i < symbols.size(); ++i) {
      if (symbols[i] == kSTAR) {
        last_star = i;
      }
    }

    size_t i = max(last_star + 1, static_cast<int>(symbols.size() - context_width));
    while (i < symbols.size()) {
      pruned_symbols.push_back(symbols[i]);
      pruned_affiliations.push_back(affiliations[i]);
      ++i;
    }
  }

  symbols = pruned_symbols;
  affiliations = pruned_affiliations;
  assert(symbols.size() == affiliations.size());
}

// Constructs the "state" of an NT, which is all the information the feature
// may want to know about a child NT when evaluating an edge.
// See comment near the top of this file to find out what all these
// fields mean.
void FF_SourceLBLLM::constructNextState(
    void* state, vector<int>& symbols, vector<int>& affiliations,
    int spanStart, int spanEnd, int targetLength,
    int leftMostLinkSource, int leftMostLinkDistance,
    int rightMostLinkSource, int rightMostLinkDistance ) const {
  prune(symbols, affiliations);
  stateConverter->convert(
      state, symbols, affiliations, targetLength, leftMostLinkSource,
      leftMostLinkDistance, rightMostLinkSource, rightMostLinkDistance,
      spanStart, spanEnd);
}

// Scores the symbols up to the first kSTAR, or up to the context_width,
// whichever is first, padding the context with kSTART or kUNKNOWN as
// needed. This offsets the fact that by scoreFullContexts() does not
// score the first context_width words of a sentence.
LBLFeatures FF_SourceLBLLM::estimateScore(
    const vector<int>& symbols, const vector<int>& affiliations) const {
  assert (symbols.size() == affiliations.size());
  LBLFeatures ret = scoreFullContexts(symbols, affiliations);

  int context_width = config->ngram_order - 1;
  for (size_t i = 0; i < symbols.size() && i < context_width; ++i) {
    if (symbols[i] == kSTAR) {
      break;
    }

    if (symbols[i] != kSTART) {
      ret += scoreContext(symbols, i, affiliations[i]);
    }
  }

  return ret;
}

FF_SourceLBLLM::~FF_SourceLBLLM() {
  savePersistentCache();
  if (persistentCache) {
    cerr << "Cache hit ratio: " << 100.0 * cacheHits / totalHits
         << " %" << endl;
  }
}

} // namespace oxlm
