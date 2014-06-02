#include "lbl/cdec_state_converter.h"

namespace oxlm {

CdecStateConverter::CdecStateConverter(int state_offset)
    : stateOffset(state_offset) {}

vector<int> CdecStateConverter::convert(const void* state) const {
  int state_size = getStateSize(state);
  const int* buffer = reinterpret_cast<const int*>(state);
  vector<int> symbols;
  for (int i = 0; i < state_size; ++i) {
    symbols.push_back(buffer[i]);
  }
  return symbols;
}

void CdecStateConverter::convert(
    const vector<int>& symbols, void* state) const {
  int* new_state = reinterpret_cast<int*>(state);
  for (size_t i = 0; i < symbols.size(); ++i) {
    new_state[i] = symbols[i];
  }
  setStateSize(state, symbols.size());
}

int CdecStateConverter::getStateSize(const void* state) const {
  return *(reinterpret_cast<const char*>(state) + stateOffset);
}

void CdecStateConverter::setStateSize(void* state, int state_size) const {
  *(reinterpret_cast<char*>(state) + stateOffset) = state_size;
}

} // namespace oxlm
