#include <cassert>
#include "lbl/cdec_state_converter.h"

namespace oxlm {

/*
These functions convert between a vector of ints and a void* "state".
The void* points to an int array followed by a single character.
The single character is at offset state_offset, and indicates
how many of the integers in the array are actually used. Any
remaining integers are garbage.

For example if the maximum state size were 4, but we wanted to
only store a vector of size 2, we would write it as:

bytes 0-4: first integer
bytes 5-8: second integer
bytes 9-12: garbage
bytes 13-16: garbage
byte 17: size of vector, in this case 2
*/

CdecStateConverter::CdecStateConverter(int state_offset)
    : stateOffset(state_offset) {}

// In the source conditional LM, a state has a bunch of extra fields
// since we have to keep track of which source words each target
// terminal is affiliated with.
const int extraConditionalInfoSize = 7;

// convert a state to a vector of ints
vector<int> CdecStateConverter::getTerminals(
    const void* state, bool hasSourceSideInfo) const {
  assert (hasSourceSideInfo);
  int state_size = getStateSize(state);
  if (hasSourceSideInfo) {
    assert (state_size >= extraConditionalInfoSize);
    state_size -= extraConditionalInfoSize;
  }
  state_size /= 2;

  const int* buffer = reinterpret_cast<const int*>(state);
  vector<int> symbols;
  for (int i = 0; i < state_size; ++i) {
    symbols.push_back(buffer[i]);
  }
  return symbols;
}

vector<int> CdecStateConverter::getAffiliations(const void* state) const {
  int state_size = getStateSize(state);
  assert (state_size >= extraConditionalInfoSize);
  state_size -= extraConditionalInfoSize;
  state_size /= 2;

  const int* buffer = reinterpret_cast<const int*>(state);
  vector<int> ret;
  for (int i = state_size; i < 2 * state_size; ++i) {
    ret.push_back(buffer[i]);
  }
  return ret;
}

// Returns the number of target words covered by this NP
int CdecStateConverter::getTargetSpanSize(const void* state) const {
  int state_size = getStateSize(state);
  const int* buffer = reinterpret_cast<const int*>(state);
  return buffer[state_size - 7];
}

// Of all the target terminals covered by this NT, look at the leftmost
// one that is aligned. Return the 
int CdecStateConverter::getLeftmostLinkSource(const void* state) const {
  int state_size = getStateSize(state);
  const int* buffer = reinterpret_cast<const int*>(state);
  return buffer[state_size - 6];
}

// Distance from left edge of this NT to the leftmost aligned target terminal.
// If the leftmost terminal is aligned, this distance is 0.
int CdecStateConverter::getLeftmostLinkDistance(const void* state) const {
  int state_size = getStateSize(state);
  const int* buffer = reinterpret_cast<const int*>(state);
  return buffer[state_size - 5];
}

int CdecStateConverter::getRightmostLinkSource(const void* state) const {
  int state_size = getStateSize(state);
  const int* buffer = reinterpret_cast<const int*>(state);
  return buffer[state_size - 4];
}

int CdecStateConverter::getRightmostLinkDistance(const void* state) const {
  int state_size = getStateSize(state);
  const int* buffer = reinterpret_cast<const int*>(state);
  return buffer[state_size - 3];
}

int CdecStateConverter::getSourceSpanStart(const void* state) const {
  int state_size = getStateSize(state);
  const int* buffer = reinterpret_cast<const int*>(state);
  return buffer[state_size - 2];
}

int CdecStateConverter::getSourceSpanEnd(const void* state) const {
  int state_size = getStateSize(state);
  const int* buffer = reinterpret_cast<const int*>(state);
  return buffer[state_size - 1];
}

void CdecStateConverter::convert(void* state, const vector<int>& terminals) const {
  const vector<int> affiliations(terminals.size(), -1);
  convert(state, terminals, affiliations, -1, -1, -1, -1, -1, -1, -1);
}

// convert an array of symbols into a state
// state is an output parameter.
void CdecStateConverter::convert(void* state,
    const vector<int>& terminals, const vector<int>& affiliations,
    int leftMostLinkSource, int leftMostLinkDistance,
    int rightMostLinkSource, int rightMostLinkDistance,
    int sourceSpanStart, int sourceSpanEnd, int targetSpanSize) const {
  bool hasConditionalExtraInfo = (sourceSpanStart != -1 && sourceSpanEnd != -1);
  assert ((!hasConditionalExtraInfo && affiliations.size() == 0) || (hasConditionalExtraInfo && affiliations.size() == terminals.size()));
  int* new_state = reinterpret_cast<int*>(state);
  size_t i = 0;
  for (int terminal : terminals) {
    new_state[i++] = terminal;
  }
  for (int affiliation : affiliations) {
    new_state[i++] = affiliation;
  }

  if (hasConditionalExtraInfo) {
    new_state[i++] = targetSpanSize;
    new_state[i++] = leftMostLinkSource;
    new_state[i++] = leftMostLinkDistance;
    new_state[i++] = rightMostLinkSource;
    new_state[i++] = rightMostLinkDistance;
    new_state[i++] = sourceSpanStart;
    new_state[i++] = sourceSpanEnd;
  }
  setStateSize(state, i);
}

int CdecStateConverter::getStateSize(const void* state) const {
  return *(reinterpret_cast<const char*>(state) + stateOffset);
}

void CdecStateConverter::setStateSize(void* state, int state_size) const {
  *(reinterpret_cast<char*>(state) + stateOffset) = state_size;
}

} // namespace oxlm
