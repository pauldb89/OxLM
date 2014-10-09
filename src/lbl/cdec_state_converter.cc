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

CdecStateConverterBase::CdecStateConverterBase(int state_offset)
    : stateOffset(state_offset) {}

int CdecStateConverterBase::getStateSize(const void* state) const {
  return *(reinterpret_cast<const char*>(state) + stateOffset);
}

void CdecStateConverterBase::setStateSize(void* state, int state_size) const {
  *(reinterpret_cast<char*>(state) + stateOffset) = state_size;
}

CdecStateConverter::CdecStateConverter(int state_offset)
    : CdecStateConverterBase(state_offset) {}

// convert a state to a vector of ints
vector<int> CdecStateConverter::getTerminals(const void* state) const {
  int state_size = getStateSize(state);
  const int* buffer = reinterpret_cast<const int*>(state);
  vector<int> symbols;
  for (int i = 0; i < state_size; ++i) {
    symbols.push_back(buffer[i]);
  }
  return symbols;
}

// convert an array of symbols into a state
// state is an output parameter.
void CdecStateConverter::convert(void* state,
    const vector<int>& terminals) const {
  int* new_state = reinterpret_cast<int*>(state);
  size_t i = 0;
  for (int terminal : terminals) {
    new_state[i++] = terminal;
  }
  setStateSize(state, i);
}

CdecConditionalStateConverter::CdecConditionalStateConverter(int state_offset)
    : CdecStateConverterBase(state_offset) {}

vector<int> CdecConditionalStateConverter::getTerminals(const void* state) const {
  int state_size = getStateSize(state);
  assert (state_size >= extraConditionalInfoSize);
  int terminal_count = (state_size - extraConditionalInfoSize / 2);

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
  int terminal_count = (state_size - extraConditionalInfoSize / 2);

  const int* buffer = reinterpret_cast<const int*>(state);
  vector<int> ret;
  for (int i = terminal_count; i < 2 * terminal_count; ++i) {
    ret.push_back(buffer[i]);
  }
  return ret;
}

// Returns the number of target words covered by this NP
int CdecConditionalStateConverter::getTargetSpanSize(const void* state) const {
  int state_size = getStateSize(state);
  const int* buffer = reinterpret_cast<const int*>(state);
  return buffer[state_size - 7];
}

// Of all the target terminals covered by this NT, look at the leftmost
// one that is aligned. Return the 
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
    int leftMostLinkSource, int leftMostLinkDistance,
    int rightMostLinkSource, int rightMostLinkDistance,
    int sourceSpanStart, int sourceSpanEnd, int targetSpanSize) const {
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
