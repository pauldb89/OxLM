#include "lbl/cdec_conditional_state_converter.h"

#include <cassert>

namespace oxlm {

CdecConditionalStateConverter::CdecConditionalStateConverter(int state_offset)
    : CdecStateConverterBase(state_offset) {}

vector<int> CdecConditionalStateConverter::getTerminals(const void* state) const {
  int state_size = getStateSize(state);
  assert (state_size >= extraConditionalInfoSize);
  int terminal_count = (state_size - extraConditionalInfoSize) / 2;

  const int* buffer = reinterpret_cast<const int*>(state);
  vector<int> ret;
  for (int i = 0; i < terminal_count; ++i) {
    ret.push_back(buffer[i]);
  }
  return ret;
}

vector<int> CdecConditionalStateConverter::getAffiliations(const void* state) const {
  int state_size = getStateSize(state);
  assert (state_size >= extraConditionalInfoSize);
  int terminal_count = (state_size - extraConditionalInfoSize) / 2;

  const int* buffer = reinterpret_cast<const int*>(state);
  vector<int> ret;
  for (int i = terminal_count; i < 2 * terminal_count; ++i) {
    ret.push_back(buffer[i]);
  }
  return ret;
}

// Returns the number of target words covered by this NT
int CdecConditionalStateConverter::getTargetSpanSize(const void* state) const {
  int state_size = getStateSize(state);
  const int* buffer = reinterpret_cast<const int*>(state);
  return buffer[state_size - 7];
}

// Of all the target terminals covered by this NT, look at the leftmost
// one that is aligned. Return the absolute source index of this terminal's
// affiliation.
int CdecConditionalStateConverter::getLeftmostLinkSource(const void* state) const {
  int state_size = getStateSize(state);
  const int* buffer = reinterpret_cast<const int*>(state);
  return buffer[state_size - 6];
}

// Distance from left edge of this NT to the leftmost aligned target terminal.
// If the leftmost terminal is aligned, this distance is 0.
int CdecConditionalStateConverter::getLeftmostLinkDistance(const void* state) const {
  int state_size = getStateSize(state);
  const int* buffer = reinterpret_cast<const int*>(state);
  return buffer[state_size - 5];
}

int CdecConditionalStateConverter::getRightmostLinkSource(const void* state) const {
  int state_size = getStateSize(state);
  const int* buffer = reinterpret_cast<const int*>(state);
  return buffer[state_size - 4];
}

int CdecConditionalStateConverter::getRightmostLinkDistance(const void* state) const {
  int state_size = getStateSize(state);
  const int* buffer = reinterpret_cast<const int*>(state);
  return buffer[state_size - 3];
}

int CdecConditionalStateConverter::getSourceSpanStart(const void* state) const {
  int state_size = getStateSize(state);
  const int* buffer = reinterpret_cast<const int*>(state);
  return buffer[state_size - 2];
}

int CdecConditionalStateConverter::getSourceSpanEnd(const void* state) const {
  int state_size = getStateSize(state);
  const int* buffer = reinterpret_cast<const int*>(state);
  return buffer[state_size - 1];
}

// convert an array of symbols into a state
// state is an output parameter.
void CdecConditionalStateConverter::convert(void* state,
    const vector<int>& terminals, const vector<int>& affiliations,
    int targetSpanSize, int leftMostLinkSource, int leftMostLinkDistance,
    int rightMostLinkSource, int rightMostLinkDistance,
    int sourceSpanStart, int sourceSpanEnd) const {
  assert (affiliations.size() == terminals.size());
  int* new_state = reinterpret_cast<int*>(state);
  size_t i = 0;
  for (int terminal : terminals) {
    new_state[i++] = terminal;
  }
  for (int affiliation : affiliations) {
    new_state[i++] = affiliation;
  }

  new_state[i++] = targetSpanSize;
  new_state[i++] = leftMostLinkSource;
  new_state[i++] = leftMostLinkDistance;
  new_state[i++] = rightMostLinkSource;
  new_state[i++] = rightMostLinkDistance;
  new_state[i++] = sourceSpanStart;
  new_state[i++] = sourceSpanEnd;

  setStateSize(state, i);
}

} // namespace oxlm
