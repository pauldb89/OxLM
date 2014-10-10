#include "lbl/cdec_state_converter_base.h"

namespace oxlm {

CdecStateConverterBase::CdecStateConverterBase(int state_offset)
    : stateOffset(state_offset) {}

int CdecStateConverterBase::getStateSize(const void* state) const {
  return *(reinterpret_cast<const char*>(state) + stateOffset);
}

void CdecStateConverterBase::setStateSize(void* state, int state_size) const {
  *(reinterpret_cast<char*>(state) + stateOffset) = state_size;
}

} // namespace oxlm
