#include "lbl/cdec_state_converter.h"

namespace oxlm {

/*
 * These functions convert between a vector of ints and a void* "state".
 * The void* points to an int array followed by a single character.
 * The single character is at offset state_offset, and indicates
 * how many of the integers in the array are actually used. Any
 * remaining integers are garbage.
 *
 * For example if the maximum state size were 4, but we wanted to
 * only store a vector of size 2, we would write it as:
 *
 * bytes 0-4: first integer
 * bytes 5-8: second integer
 * bytes 9-12: garbage
 * bytes 13-16: garbage
 * byte 17: size of vector, in this case 2
*/

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

} // namespace oxlm
